"""Rinha de seis CNNs: três sementes, tempo/RAM/VRAM, relatório e explicabilidade.
Usa Treino/Validacao/Teste existentes. Não redistribui nem altera imagens.
python rinha_modelos_v5.py --verificar
python rinha_modelos_v5.py
python rinha_modelos_v5.py --retomar "PASTA_DA_EXECUCAO_V5"
python rinha_modelos_v5.py --so-relatorio "PASTA_DA_EXECUCAO_V5"
python rinha_modelos_v5.py --so-explicabilidade "PASTA_DA_EXECUCAO_V5"
Dependências: torch torchvision numpy pillow matplotlib scikit-learn psutil lime shap scikit-image
"""
import argparse
import base64
import csv
import gc
import hashlib
import html
import importlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
    cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader
from torchvision import models, transforms

BASE = Path(r'C:\Users\rammo\OneDrive\Desktop\Mestrado\Modelagem')
MODELS = ['mobilenet_v2', 'mobilenet_v3_small', 'shufflenet_v2_x1_0',
          'efficientnet_b0', 'resnet18', 'densenet121']
SEEDS = [42, 123, 2026]
NAMES = ['NonPsor', 'PSOR']
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SIZE = 224
BATCH = 16
EPOCHS = 20
PATIENCE = 3
LR = 0.001
THRESHOLD = 0.5
XAI_SEED = 42
XAI_PER_CLASS = 2
LIME_SAMPLES = 1000
SHAP_EVALS = 500
BENCH_REPEATS = 3
WARMUP = 10
MEM_INTERVAL = 0.05
THREADS = min(4, os.cpu_count() or 1)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VERSION = '5.0'
DISPLAY = {'mobilenet_v2': 'MobileNetV2', 'mobilenet_v3_small': 'MobileNetV3 Small',
    'shufflenet_v2_x1_0': 'ShuffleNetV2 x1.0', 'efficientnet_b0': 'EfficientNet-B0',
    'resnet18': 'ResNet18', 'densenet121': 'DenseNet121'}
COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#56B4E9']


def configure(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(THREADS)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sync():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()


def dump(path, value):
    """Escrita atômica: marcadores de conclusão não ficam parcialmente escritos."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter=';')
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter=';'))


class Meter:
    """RSS amostrado do processo; VRAM do alocador PyTorch; nunca estima energia."""
    def __init__(self, phase, **metadata):
        self.phase = phase
        self.metadata = metadata
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.samples = []

    def sample(self):
        self.samples.append(self.process.memory_info().rss / 1024**2)

    def watch(self):
        while not self.stop.wait(MEM_INTERVAL):
            self.sample()

    def __enter__(self):
        sync()
        self.sample()
        self.initial = self.samples[0]
        self.cpu_start = self.process.cpu_times()
        if DEVICE.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        sync()
        self.elapsed = time.perf_counter() - self.start
        self.stop.set()
        self.thread.join()
        self.sample()
        cpu_end = self.process.cpu_times()
        cpu_s = (cpu_end.user + cpu_end.system) - (self.cpu_start.user + self.cpu_start.system)
        self.result = dict(self.metadata, Fase=self.phase, Tempo_s=self.elapsed,
            CPU_tempo_s=max(0, cpu_s), CPU_equivalente_percentual=100*max(0, cpu_s)/max(self.elapsed, 1e-9),
            RAM_RSS_inicial_MiB=self.initial, RAM_RSS_pico_MiB=max(self.samples),
            RAM_RSS_media_amostral_MiB=float(np.mean(self.samples)),
            RAM_RSS_final_MiB=self.samples[-1], RAM_incremento_pico_MiB=max(self.samples)-self.initial,
            RAM_amostras=len(self.samples), Intervalo_alvo_s=MEM_INTERVAL,
            GPU_pico_alocado_MiB=(torch.cuda.max_memory_allocated()/1024**2 if DEVICE.type == 'cuda' else None),
            GPU_pico_reservado_MiB=(torch.cuda.max_memory_reserved()/1024**2 if DEVICE.type == 'cuda' else None))


def architecture(name, pretrained=True):
    builders = {
        'mobilenet_v2': (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V2),
        'mobilenet_v3_small': (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
        'shufflenet_v2_x1_0': (models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
        'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        'densenet121': (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
    }
    constructor, weights = builders[name]
    model = constructor(weights=weights if pretrained else None)
    if name in ['resnet18', 'shufflenet_v2_x1_0']:
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif name == 'densenet121':
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    else:
        last = len(model.classifier) - 1
        model.classifier[last] = nn.Linear(model.classifier[last].in_features, 1)
    return model.to(DEVICE)


def datasets(base):
    normal = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    eval_t = transforms.Compose([transforms.Resize((SIZE, SIZE))] + normal)
    train_t = transforms.Compose([transforms.Resize((SIZE, SIZE)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)] + normal)
    data = {split: torchvision.datasets.ImageFolder(Path(base)/split,
        transform=train_t if split == 'Treino' else eval_t) for split in ['Treino', 'Validacao', 'Teste']}
    for split, ds in data.items():
        if ds.class_to_idx != {'NonPsor': 0, 'PSOR': 1} or set(ds.targets) != {0, 1}:
            raise ValueError(f'{split}: devem existir somente as classes NonPsor e PSOR, ambas preenchidas.')
    return data


def get_loader(ds, seed=42, train=False):
    return DataLoader(ds, batch_size=BATCH if train else 1, shuffle=train, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def audit(data, output=None):
    seen, manifest, duplicates, counts = {}, [], [], []
    for split, ds in data.items():
        for label, count in sorted(Counter(ds.targets).items()):
            counts.append({'Conjunto': split, 'Classe': NAMES[label], 'Quantidade': count})
        for path, label in ds.samples:
            with Image.open(path) as image:
                image = image.convert('RGB')
                image.load()
                digest = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
            row = {'Conjunto': split, 'Classe': NAMES[label], 'Arquivo': str(Path(path).resolve()), 'Hash_pixels': digest}
            if digest in seen and (seen[digest]['Conjunto'] != split or seen[digest]['Classe'] != NAMES[label]):
                duplicates.append({'Arquivo_1': seen[digest]['Arquivo'], 'Arquivo_2': row['Arquivo']})
            seen.setdefault(digest, row)
            manifest.append(row)
    if output:
        write_csv(output/'csv'/'manifesto_imagens.csv', manifest)
        write_csv(output/'csv'/'distribuicao_imagens.csv', counts)
        write_csv(output/'csv'/'duplicatas_conflitantes.csv', duplicates)
    if duplicates:
        raise ValueError(f'{len(duplicates)} duplicatas exatas entre conjuntos/classes. Corrija antes de executar.')
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(), counts


def metrics(y, predicted, p):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {'Accuracy': float(accuracy_score(y, predicted)),
        'Balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
        'F1': float(f1_score(y, predicted, zero_division=0)),
        'F1_macro': float(f1_score(y, predicted, average='macro', zero_division=0)),
        'Recall': float(recall_score(y, predicted, zero_division=0)),
        'Specificity': float(tn/(tn+fp)), 'Precision': float(precision_score(y, predicted, zero_division=0)),
        'AUC': float(roc_auc_score(y, p)), 'Kappa': float(cohen_kappa_score(y, predicted)),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp), 'N': len(y)}


def paths(output, name, seed):
    stem = f'{name}_s{seed}'
    return stem, output/'checkpoints'/f'{stem}.pt', output/'dados'/f'{stem}_treino.json', output/'dados'/f'{stem}_avaliacao.json'


def train_worker(output, name, seed):
    config = read(output/'configuracao.json')
    data = datasets(config['base'])
    stem, checkpoint, trained, evaluated = paths(output, name, seed)
    resources, history = [], []
    with Meter('Inicializacao_pre_treinada', Modelo=name, Semente=seed) as meter:
        model = architecture(name)
    resources.append(meter.result)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    train_dl = get_loader(data['Treino'], seed, True)
    val_dl = get_loader(data['Validacao'], seed)
    best, best_epoch, waiting = float('inf'), 0, 0
    sync()
    start = time.perf_counter()
    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss = 0.0
        with Meter('Treino_epoca', Modelo=name, Semente=seed, Epoca=epoch) as train_meter:
            for images, labels in train_dl:
                labels = labels.float().view(-1, 1).to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(images.to(DEVICE)), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Perda de treino não finita: {name}, época {epoch}')
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*len(labels)
        resources.append(train_meter.result)
        model.eval()
        val_loss = 0.0
        with Meter('Validacao_epoca', Modelo=name, Semente=seed, Epoca=epoch) as val_meter:
            with torch.no_grad():
                for images, labels in val_dl:
                    labels = labels.float().view(-1, 1).to(DEVICE)
                    val_loss += loss_fn(model(images.to(DEVICE)), labels).item()*len(labels)
        resources.append(val_meter.result)
        val_loss /= len(data['Validacao'])
        if not np.isfinite(val_loss):
            raise RuntimeError('Perda de validação não finita.')
        history.append({'Modelo': name, 'Semente': seed, 'Epoca': epoch,
            'Train_loss': train_loss/len(data['Treino']), 'Val_loss': val_loss,
            'Tempo_treino_s': train_meter.elapsed, 'Tempo_validacao_s': val_meter.elapsed})
        write_csv(output/'csv'/'historicos'/f'{stem}.csv', history)
        write_csv(output/'csv'/'recursos'/f'{stem}_treino.csv', resources)
        print(f'{name} | semente {seed} | época {epoch}/{EPOCHS} | val_loss={val_loss:.5f}', flush=True)
        if val_loss < best:
            best, best_epoch, waiting = val_loss, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            waiting += 1
            if waiting >= PATIENCE:
                break
    sync()
    train_stages = [row for row in resources if row['Fase']=='Treino_epoca']
    val_stages = [row for row in resources if row['Fase']=='Validacao_epoca']
    dump(trained, {'Modelo': name, 'Semente': seed, 'Epocas': epoch, 'Melhor_epoca': best_epoch,
        'Melhor_val_loss': best, 'Treino_validacao_total_s': time.perf_counter()-start,
        'Treino_epocas_s': sum(row['Tempo_s'] for row in train_stages),
        'Validacao_epocas_s': sum(row['Tempo_s'] for row in val_stages),
        'Treino_RAM_pico_MiB': max(row['RAM_RSS_pico_MiB'] for row in train_stages),
        'Treino_GPU_pico_MiB': max((row['GPU_pico_alocado_MiB'] or 0) for row in train_stages) if DEVICE.type=='cuda' else None,
        'Parametros': sum(p.numel() for p in model.parameters()),
        'Pesos_buffers_MiB': sum(t.numel()*t.element_size() for t in model.state_dict().values())/1024**2,
        'Checkpoint_MiB': checkpoint.stat().st_size/1024**2})


def load_model(name, checkpoint):
    model = architecture(name, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    return model.eval()


def eval_worker(output, name, seed):
    data = datasets(read(output/'configuracao.json')['base'])
    stem, checkpoint, trained, evaluated = paths(output, name, seed)
    result = read(trained)
    resources = []
    with Meter('Carregamento_checkpoint', Modelo=name, Semente=seed) as meter:
        model = load_model(name, checkpoint)
    resources.append(meter.result)
    # Processo novo, sem otimizador e sem tensores do treinamento.
    with torch.inference_mode():
        warm = next(iter(get_loader(data['Teste'])))[0].to(DEVICE)
        for _ in range(WARMUP):
            torch.sigmoid(model(warm))
        del warm
    sync()
    for split, short in [('Validacao','Val'), ('Teste','Test')]:
        y, p = [], []
        with Meter(f'Avaliacao_{short}_com_IO', Modelo=name, Semente=seed) as meter:
            with torch.inference_mode():
                for images, labels in get_loader(data[split]):
                    prob = torch.sigmoid(model(images.to(DEVICE))).flatten().cpu().tolist()
                    p.extend(prob)
                    y.extend(labels.tolist())
        resources.append(meter.result)
        predicted = (np.asarray(p)>=THRESHOLD).astype(int).tolist()
        calculated = metrics(y, predicted, p)
        result.update({f'{short}_{key}': value for key, value in calculated.items()})
        result[f'{short}_avaliacao_total_s'] = meter.elapsed
        result[f'{short}_avaliacao_ms_imagem'] = 1000*meter.elapsed/len(y)
        result[f'{short}_RAM_pico_MiB'] = meter.result['RAM_RSS_pico_MiB']
        write_csv(output/'csv'/'predicoes'/f'{stem}_{short}.csv', [
            {'Arquivo': data[split].samples[i][0], 'Real': y[i], 'Predita': predicted[i],
             'P_PSOR': p[i]} for i in range(len(y))])
        write_csv(output/'csv'/'matrizes'/f'{stem}_{short}.csv', [
            {'Classe_real':'NonPsor','Pred_NonPsor':calculated['TN'],'Pred_PSOR':calculated['FP']},
            {'Classe_real':'PSOR','Pred_NonPsor':calculated['FN'],'Pred_PSOR':calculated['TP']}])
        (output/'dados'/f'{stem}_{short}_classificacao.txt').write_text(
            classification_report(y, predicted, labels=[0,1], target_names=NAMES, zero_division=0), encoding='utf-8')
    # Latência de lote unitário; relógio exclui leitura, transformação e transferência.
    latency = []
    with Meter('Benchmark_forward', Modelo=name, Semente=seed) as meter:
        with torch.inference_mode():
            for repeat in range(BENCH_REPEATS):
                for index, (images, _) in enumerate(get_loader(data['Teste'])):
                    images = images.to(DEVICE)
                    sync()
                    start = time.perf_counter()
                    torch.sigmoid(model(images))
                    sync()
                    latency.append({'Repeticao':repeat+1, 'Imagem_indice':index,
                                    'Latencia_ms':1000*(time.perf_counter()-start)})
    resources.append(meter.result)
    values = np.asarray([row['Latencia_ms'] for row in latency])
    result.update({'Forward_media_ms':float(values.mean()), 'Forward_mediana_ms':float(np.median(values)),
        'Forward_p95_ms':float(np.percentile(values,95)), 'Forward_desvio_ms':float(values.std(ddof=1)),
        'Forward_imagens_s':1000/float(values.mean()), 'Forward_medicoes':len(values),
        'Inferencia_RAM_inicial_MiB':meter.result['RAM_RSS_inicial_MiB'],
        'Inferencia_RAM_pico_MiB':meter.result['RAM_RSS_pico_MiB'],
        'Inferencia_RAM_incremento_MiB':meter.result['RAM_incremento_pico_MiB'],
        'Inferencia_GPU_pico_MiB':meter.result['GPU_pico_alocado_MiB']})
    write_csv(output/'csv'/'latencias'/f'{stem}.csv', latency)
    write_csv(output/'csv'/'recursos'/f'{stem}_avaliacao.csv', resources)
    dump(evaluated, result)


def gradcam(model, name, tensor, target=1):
    if name=='resnet18': layer = model.layer4
    elif name=='shufflenet_v2_x1_0': layer = model.conv5
    elif name=='densenet121': layer = model.features.denseblock4
    else: layer = model.features[-1]
    captured = []
    handle = layer.register_forward_hook(lambda module, inp, out: captured.append(out))
    try:
        with torch.enable_grad():
            logits = model(tensor.detach().clone().requires_grad_(True)).flatten()
            score = logits[0] if target==1 else -logits[0]
            gradients = torch.autograd.grad(score, captured[0])[0]
            raw = torch.relu((gradients.mean((2,3),keepdim=True)*captured[0]).sum(1,keepdim=True))
            raw = F.interpolate(raw,size=(SIZE,SIZE),mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy()
        amplitude = float(np.ptp(raw))
        cam = (raw-raw.min())/amplitude if amplitude>1e-12 else np.zeros_like(raw)
        if not np.isfinite(cam).all(): raise ValueError('Grad-CAM não finito.')
        return cam, amplitude
    finally:
        handle.remove()


def optional_tools(required=True):
    tools, errors = {}, []
    for name, module in [('LIME','lime.lime_image'),('SHAP','shap')]:
        try: tools[name] = importlib.import_module(module)
        except Exception as exc: errors.append(f'{name}: {exc}')
    if errors and required:
        raise RuntimeError('Instale/corrija antes de iniciar: python -m pip install lime shap scikit-image\n'+'\n'.join(errors))
    return tools


def rgb_predictor(model):
    mean = torch.tensor(MEAN,device=DEVICE).view(1,3,1,1)
    std = torch.tensor(STD,device=DEVICE).view(1,3,1,1)
    def predict(images):
        array = np.asarray(images,dtype=np.float32)
        results = []
        with torch.inference_mode():
            for offset in range(0,len(array),BATCH):
                x = torch.from_numpy(np.ascontiguousarray(array[offset:offset+BATCH])).permute(0,3,1,2).to(DEVICE)
                p = torch.sigmoid(model((x-mean)/std)).flatten()
                results.append(torch.stack([1-p,p],1).cpu().numpy())
        return np.concatenate(results)
    return predict


def lime_map(tools, rgb, predict, samples=LIME_SAMPLES):
    from skimage.segmentation import slic
    result = tools['LIME'].LimeImageExplainer(random_state=XAI_SEED).explain_instance(
        rgb.astype(np.float64), predict, labels=(1,), top_labels=None, num_samples=samples,
        batch_size=BATCH, segmentation_fn=lambda img:slic(img,n_segments=50,compactness=10,start_label=0),
        random_seed=XAI_SEED)
    values = np.zeros((SIZE,SIZE),dtype=np.float32)
    for segment, weight in result.local_exp[1]: values[result.segments==segment]=weight
    return values, float(result.score), int(len(np.unique(result.segments)))


def shap_map(tools, rgb, predict, evaluations=SHAP_EVALS):
    shap = tools['SHAP']
    masker = shap.maskers.Image(np.zeros_like(rgb),rgb.shape)
    explainer = shap.Explainer(predict,masker,algorithm='partition',output_names=NAMES)
    result = explainer(rgb[None],max_evals=evaluations,batch_size=BATCH,outputs=[1])
    values = np.asarray(result.values)
    if values.shape!=(1,SIZE,SIZE,3,1): raise ValueError(f'Formato SHAP inesperado: {values.shape}')
    return values[0,...,0].sum(-1), values, float(np.asarray(result.base_values).reshape(-1)[0])


def colored_overlay(rgb, values, signed=False):
    if not np.isfinite(values).all(): raise ValueError('Mapa contém valores não finitos.')
    if signed:
        bound = max(float(np.max(np.abs(values))),1e-12)
        color = plt.get_cmap('RdBu_r')((values/bound+1)/2)[...,:3]
        alpha = 0.55*np.abs(values)[...,None]/bound
    else:
        color = plt.get_cmap('inferno')(values)[...,:3]
        alpha = 0.55*values[...,None]
    return np.clip(rgb*(1-alpha)+color*alpha,0,1), color


def xai_worker(output):
    config = read(output/'configuracao.json')
    choice = read(output/'modelo_selecionado.json')
    name, seed = choice['Modelo'], choice['Semente']
    model = load_model(name,paths(output,name,seed)[1])
    tools = optional_tools()
    data = datasets(config['base'])['Teste']
    predict = rgb_predictor(model)
    indices, rng = [], random.Random(XAI_SEED)
    for label in [0,1]:
        candidates = [i for i,y in enumerate(data.targets) if y==label]
        indices.extend(rng.sample(candidates,min(XAI_PER_CLASS,len(candidates))))
    folder = output/'png'/'explicabilidade'
    folder.mkdir(parents=True,exist_ok=True)
    resources, status, figures = [], [], []
    for order,index in enumerate(indices,1):
        tensor,label = data[index]
        rgb = (tensor.permute(1,2,0).numpy()*np.asarray(STD)+np.asarray(MEAN)).clip(0,1).astype(np.float32)
        p = float(predict(rgb[None])[0,1])
        predicted = int(p>=THRESHOLD)
        prefix = f'{order:02d}_{NAMES[label]}'
        plt.imsave(folder/f'{prefix}_original.png',rgb)
        panels = {'Original':rgb}
        captions = {'Original':f'Real: {NAMES[label]}\nPredita: {NAMES[predicted]} | P(PSOR)={p:.3f}'}
        for method in ['LIME','SHAP','GradCAM']:
            meter = None
            record = {'ID':prefix,'Arquivo':data.samples[index][0],'Modelo':name,'Semente':seed,
                'Classe_real':NAMES[label],'Classe_predita':NAMES[predicted], 'Classe_explicada':'PSOR',
                'P_PSOR':p,'Metodo':method,'Status':'','R2_local':None,'Detalhe':''}
            try:
                with Meter(f'XAI_{method}',Modelo=name,Semente=seed,Imagem=prefix) as meter:
                    if method=='GradCAM':
                        values,amplitude=gradcam(model,name,tensor[None].to(DEVICE),1)
                        record['Status']='OK' if amplitude>1e-12 else 'MAPA_SEM_ATIVACAO'
                        record['Detalhe']=f'Amplitude antes de normalizar: {amplitude:.6g}'
                        captions[method]='Contribuição positiva para PSOR\n'+('Mapa não nulo' if amplitude>1e-12 else 'Sem ativação positiva variável')
                    elif method=='LIME':
                        values,r2,n_segments=lime_map(tools,rgb,predict,samples=LIME_SAMPLES)
                        record.update(Status='OK',R2_local=r2,Detalhe=f'{LIME_SAMPLES} perturbações; {n_segments} segmentos')
                        captions[method]=f'Aproximação local | R²={r2:.3f}\n{LIME_SAMPLES} perturbações'
                    else:
                        values,channels,base_value=shap_map(tools,rgb,predict,evaluations=SHAP_EVALS)
                        np.save(output/'dados'/f'{prefix}_shap_canais.npy',channels)
                        record.update(Status='OK',Detalhe=f'Partition; referência preta; max_evals={SHAP_EVALS}; base={base_value:.5f}')
                        captions[method]=f'Contribuições relativas à referência preta\nLimite: {SHAP_EVALS} avaliações'
                    overlay,color=colored_overlay(rgb,values,signed=method!='GradCAM')
                panels[method]=overlay
                np.save(output/'dados'/f'{prefix}_{method}_valores.npy',values)
                plt.imsave(folder/f'{prefix}_{method}_sobreposicao.png',overlay)
                plt.imsave(folder/f'{prefix}_{method}_mapa.png',color)
            except Exception as exc:
                record.update(Status='ERRO',Detalhe=str(exc))
                captions[method]='Falha registrada no log'
                with (output/'logs'/'explicabilidade_erros.txt').open('a',encoding='utf-8') as f:
                    f.write(f'\n{prefix} {method}\n{traceback.format_exc()}')
            if meter is not None and hasattr(meter,'result'): resources.append(meter.result)
            status.append(record)
            write_csv(output/'csv'/'status_explicabilidade.csv',status)
            write_csv(output/'csv'/'recursos_explicabilidade.csv',resources)
            print(prefix,method,record['Status'],record['Detalhe'],flush=True)
        figures.append((prefix,panels,captions))
        xai_panel([(prefix,panels,captions)],folder/f'{prefix}_comparativo.png',name)
    xai_panel(figures,output/'png'/'03_explicabilidade.png',name)
    dump(output/'dados'/'explicabilidade_concluida.json',dict(Counter(r['Status'] for r in status)))


def xai_panel(figures,path,name):
    fig,axes=plt.subplots(len(figures),4,figsize=(14,4.8*len(figures)+0.5),squeeze=False)
    for i,(prefix,panels,captions) in enumerate(figures):
        for j,method in enumerate(['Original','LIME','SHAP','GradCAM']):
            ax=axes[i,j]
            if method in panels: ax.imshow(panels[method])
            else: ax.text(.5,.5,'Indisponível',ha='center',va='center',transform=ax.transAxes)
            ax.set_title(('Grad-CAM' if method=='GradCAM' else method),fontweight='bold')
            ax.set_xticks([]);ax.set_yticks([])
            ax.set_xlabel(captions.get(method,''),fontsize=9)
            if j==0: ax.set_ylabel(prefix,fontsize=10)
    fig.suptitle(f'{DISPLAY[name]} — alvo fixo: PSOR',fontsize=15,fontweight='bold')
    fig.text(.5,.01,'LIME/SHAP: vermelho positivo, azul negativo. Grad-CAM: intensidade positiva. Escalas independentes por mapa.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96), h_pad=3.5, w_pad=1.0)
    fig.savefig(path,dpi=220)
    plt.close(fig)


def collect(output):
    rows=[]
    for name in MODELS:
        for seed in SEEDS:
            path=paths(output,name,seed)[3]
            if path.exists(): rows.append(read(path))
    summaries=[]
    for name in MODELS:
        group=[r for r in rows if r['Modelo']==name]
        if not group: continue
        summary={'Modelo':name,'Execucoes':len(group),'Completo':len(group)==len(SEEDS)}
        keys=[k for k,v in group[0].items() if isinstance(v,(int,float)) and k!='Semente']
        for key in keys:
            values=[r[key] for r in group if isinstance(r.get(key),(int,float))]
            summary[key+'_media']=float(np.mean(values))
            summary[key+'_dp']=float(np.std(values,ddof=1)) if len(values)>1 else None
        summaries.append(summary)
    return rows,summaries


def select(output,rows,summaries):
    eligible=[s for s in summaries if s['Completo']]
    if not eligible: return None
    selected=sorted(eligible,key=lambda s:(-s['Val_F1_media'],-s['Val_AUC_media'],s['Modelo']))[0]
    representatives=[]
    for summary in summaries:
        group=[r for r in rows if r['Modelo']==summary['Modelo']]
        representative=sorted(group,key=lambda r:(-r['Val_F1'],-r['Val_AUC'],r['Semente']))[0]
        representatives.append(representative)
    representative=next(r for r in representatives if r['Modelo']==selected['Modelo'])
    choice={'Modelo':representative['Modelo'],'Semente':representative['Semente'],
        'Criterio_arquitetura':'Maior F1 médio de validação nas três sementes; desempate por AUC média de validação.',
        'Criterio_checkpoint':'Melhor F1 de validação da arquitetura escolhida; desempate por AUC e semente.',
        'Cobertura':f'{len(rows)}/{len(MODELS)*len(SEEDS)} execuções concluídas'}
    dump(output/'modelo_selecionado.json',choice)
    write_csv(output/'csv'/'checkpoints_representativos_validacao.csv',representatives)
    return choice


def save_plot(fig,path):
    fig.tight_layout()
    fig.savefig(path,dpi=240,bbox_inches='tight')
    plt.close(fig)


def charts(output,rows,summaries):
    if not summaries:return
    fig,ax=plt.subplots(figsize=(9,6))
    for i,s in enumerate(summaries):
        ax.errorbar(s['Forward_mediana_ms_media'],s['Test_F1_media'],
            xerr=s.get('Forward_mediana_ms_dp') or 0,yerr=s.get('Test_F1_dp') or 0,
            fmt='o',color=COLORS[i],capsize=4,label=DISPLAY[s['Modelo']])
    ax.set(xlabel='Latência mediana da rede (ms/imagem): média entre sementes',
        ylabel='F1 da psoríase no teste: média ± DP',title='Desempenho e tempo de execução')
    ax.grid(alpha=.2);ax.legend(fontsize=9)
    save_plot(fig,output/'png'/'01_desempenho_tempo.png')
    representatives=[]
    for s in summaries:
        representatives.append(sorted([r for r in rows if r['Modelo']==s['Modelo']],
            key=lambda r:(-r['Val_F1'],-r['Val_AUC'],r['Semente']))[0])
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for ax,row in zip(axes.flat,representatives):
        matrix=np.asarray([[row['Test_TN'],row['Test_FP']],[row['Test_FN'],row['Test_TP']]])
        percentages=100*matrix/matrix.sum(1,keepdims=True)
        ax.imshow(percentages,cmap='Blues',vmin=0,vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f'{matrix[i,j]}\n{percentages[i,j]:.1f}%',ha='center',va='center',
                        color='white' if percentages[i,j]>55 else 'black',fontsize=12)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=NAMES,yticklabels=NAMES,
            ylabel='Classe real',title=f"{DISPLAY[row['Modelo']]} | semente {row['Semente']}")
        ax.set_xlabel(f"Classe predita\nPSOR: {row['Test_TP']}/{row['Test_TP']+row['Test_FN']} identificadas; "
            f"falsos positivos: {row['Test_FP']}/{row['Test_TN']+row['Test_FP']}",fontsize=9)
    for ax in list(axes.flat)[len(representatives):]:ax.axis('off')
    fig.suptitle('Matrizes de teste — checkpoint de cada arquitetura selecionado pela validação\nContagens e percentuais por classe real; escala comum de 0 a 100%',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(output/'png'/'02_matrizes_confusao.png',dpi=240,bbox_inches='tight');plt.close(fig)
    labels=[DISPLAY[s['Modelo']] for s in summaries]
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    specs=[('Treino_validacao_total_s','Treinamento com validação e checkpoints (s)'),
           ('Forward_mediana_ms','Latência mediana da rede (ms)'),
           ('Treino_RAM_pico_MiB','Pico de RAM no treino (MiB de RSS do processo)'),
           ('Inferencia_RAM_pico_MiB','Pico de RAM na inferência (MiB de RSS do processo)')]
    for ax,(key,title) in zip(axes.flat,specs):
        values=[s[key+'_media'] for s in summaries]
        errors=[s.get(key+'_dp') or 0 for s in summaries]
        ax.barh(labels,values,xerr=errors,color=COLORS[:len(summaries)],capsize=3)
        ax.invert_yaxis();ax.set_title(title,fontsize=11);ax.set_xlabel('Média ± DP entre sementes')
    save_plot(fig,output/'png'/'04_recursos_computacionais.png')
    fig,ax=plt.subplots(figsize=(7,6))
    for i,row in enumerate(representatives):
        stem=paths(output,row['Modelo'],row['Semente'])[0]
        pred=read_csv(output/'csv'/'predicoes'/f'{stem}_Test.csv')
        y=[int(r['Real']) for r in pred];p=[float(r['P_PSOR']) for r in pred]
        fpr,tpr,_=roc_curve(y,p)
        ax.plot(fpr,tpr,color=COLORS[i],label=f"{DISPLAY[row['Modelo']]}: {row['Test_AUC']:.3f}")
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(xlabel='Taxa de falsos positivos',ylabel='Sensibilidade',title='ROC de teste — checkpoints escolhidos pela validação')
    ax.legend(fontsize=9)
    save_plot(fig,output/'png'/'05_curvas_roc.png')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,summary in zip(axes.flat,summaries):
        name=summary['Modelo']
        for i,seed in enumerate(SEEDS):
            history=read_csv(output/'csv'/'historicos'/f'{name}_s{seed}.csv')
            if history:
                ax.plot([int(r['Epoca']) for r in history],[float(r['Val_loss']) for r in history],
                        color=COLORS[i],label=f'Validação: {seed}')
        ax.set(title=DISPLAY[name],xlabel='Época',ylabel='Perda de validação');ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(summaries):]:ax.axis('off')
    save_plot(fig,output/'png'/'06_historicos_validacao.png')


def number(value,digits=3):
    if value is None:return '—'
    if isinstance(value,(int,float)):
        return f'{value:.{digits}f}'.replace('.',',')
    return str(value)


def mean_sd(s,key,digits=3):
    return number(s.get(key+'_media'),digits)+' ± '+number(s.get(key+'_dp'),digits)


def html_table(headers,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def report(output):
    config=read(output/'configuracao.json')
    rows,summaries=collect(output)
    write_csv(output/'csv'/'resultados_por_execucao.csv',rows)
    write_csv(output/'csv'/'resumo_por_arquitetura.csv',summaries)
    choice=select(output,rows,summaries)
    charts(output,rows,summaries)
    all_resources=[]
    for path in sorted((output/'csv'/'recursos').glob('*.csv')):all_resources.extend(read_csv(path))
    write_csv(output/'csv'/'recursos_por_etapa.csv',all_resources)
    expected=len(MODELS)*len(SEEDS)
    complete=len(rows)==expected
    pieces=['<h1>Comparação de arquiteturas para classificação de psoríase</h1>',
        '<p class="intro">Desempenho preditivo, tempo de execução, memória e explicabilidade.</p>',
        f'<p><b>Execuções concluídas:</b> {len(rows)}/{expected} · <b>Dispositivo:</b> {html.escape(config["device"])} · '
        f'<b>Sementes:</b> {html.escape(str(SEEDS))}</p>']
    if not complete:
        pieces.append('<p class="notice">Relatório parcial. Consulte csv/status_execucoes.csv e logs. A seleção, quando disponível, considera somente arquiteturas com as três sementes concluídas.</p>')
    if choice:
        pieces.append(f'<p class="card"><b>Arquitetura selecionada pela validação:</b> {DISPLAY[choice["Modelo"]]}.'
            f' Checkpoint representativo: semente {choice["Semente"]}. O critério considera F1 médio de validação, com AUC média como desempate; custo computacional é apresentado separadamente.</p>')
    pieces.append('<h2>1. Dados e protocolo</h2>')
    count_rows=read_csv(output/'csv'/'distribuicao_imagens.csv')
    pieces.append(html_table(['Conjunto','Classe','Imagens'],[[r['Conjunto'],r['Classe'],r['Quantidade']] for r in count_rows]))
    pieces.append(f'<p>Partição fornecida pelo usuário, preservada. Entrada {SIZE} × {SIZE}; normalização ImageNet; aumento de dados somente no treino; '
        f'fine-tuning integral, Adam, taxa {LR}, lote {BATCH}, máximo {EPOCHS} épocas, paciência {PATIENCE}, limiar {THRESHOLD}. '
        f'Cada arquitetura usa pesos ImageNet explicitados no script. Execuções independentes por semente; avaliação em novo processo sem otimizador. '
        f'CPU configurada para {THREADS} threads do PyTorch. Pesos de pré-treinamento podem ter receitas distintas; a comparação é entre implementações pré-treinadas sob um protocolo comum.</p>')
    pieces.append('<h2>2. Desempenho no teste</h2><p>Média ± desvio-padrão entre sementes. O desvio mede variabilidade do treinamento na partição fixa; não é intervalo de confiança nem validação externa.</p>')
    pieces.append(html_table(['Arquitetura','n','F1 PSOR','Sensibilidade','Especificidade','AUC','Acurácia','Kappa'],
        [[DISPLAY[s['Modelo']],s['Execucoes']]+[mean_sd(s,k) for k in ['Test_F1','Test_Recall','Test_Specificity','Test_AUC','Test_Accuracy','Test_Kappa']] for s in summaries]))
    pieces.append('<p>Precisão, F1 macro, acurácia balanceada, métricas de validação, TN/FP/FN/TP e resultados individuais estão nos CSVs. Nenhuma hipótese de superioridade estatística foi testada.</p>')
    pieces.append('<h2>3. Custo computacional</h2>')
    pieces.append(html_table(['Arquitetura','Parâmetros (milhões)','Checkpoint (MiB)','Treino + validação (s)','Rede: mediana (ms)','Rede: p95 (ms)','RAM pico treino (MiB)','RAM pico inferência (MiB)'],
        [[DISPLAY[s['Modelo']],number(s['Parametros_media']/1e6,2),number(s['Checkpoint_MiB_media'],2),
          mean_sd(s,'Treino_validacao_total_s',1),mean_sd(s,'Forward_mediana_ms',2),mean_sd(s,'Forward_p95_ms',2),
          mean_sd(s,'Treino_RAM_pico_MiB',1),mean_sd(s,'Inferencia_RAM_pico_MiB',1)] for s in summaries]))
    pieces.append('<p><b>Tempo de treino:</b> inclui épocas, validação, checkpoints e registros; inicialização/download dos pesos é medido separadamente. '
        '<b>Avaliação com I/O:</b> leitura, transformação, transferência, forward e coleta das saídas; cálculo final de métricas fora do relógio. '
        '<b>Latência da rede:</b> forward e sigmoide, lote unitário, entrada já transferida, aquecimento prévio, três passagens; não é latência de aplicativo móvel. '
        '<b>RAM:</b> RSS do processo, incluindo intérprete, bibliotecas, tensores e alocações mantidas; não corresponde somente aos pesos. '
        'RSS é amostrado a cada 50 ms aproximadamente, podendo perder picos breves. CSVs registram nível inicial, pico, incremento, média amostral e final. '
        'GPU alocada/reservada é medida pelo alocador PyTorch quando CUDA está disponível; campos ficam vazios em CPU. '
        'CPU_tempo_s acumula tempo de CPU; CPU_equivalente_percentual pode superar 100% em uso de múltiplos núcleos. '
        '<b>Energia elétrica não é medida.</b> Tempo de execução não equivale a consumo energético.</p>')
    pieces.append('<h2>4. Figuras de síntese</h2>')
    for filename,title in [('01_desempenho_tempo.png','Qualidade classificatória e latência'),
        ('02_matrizes_confusao.png','Matrizes de confusão: um checkpoint por arquitetura'),
        ('04_recursos_computacionais.png','Tempo e memória'),('05_curvas_roc.png','Curvas ROC'),
        ('06_historicos_validacao.png','Evolução da validação')]:
        path=output/'png'/filename
        if path.exists():
            encoded=base64.b64encode(path.read_bytes()).decode()
            pieces.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(title)}"><figcaption>{html.escape(title)}</figcaption></figure>')
    pieces.append('<h2>5. Explicabilidade</h2>')
    statuses=read_csv(output/'csv'/'status_explicabilidade.csv')
    if statuses and choice and any(r['Modelo']!=choice['Modelo'] or int(r['Semente'])!=choice['Semente'] for r in statuses):
        pieces.append('<p>Os mapas existentes pertencem a outro checkpoint. Execute --so-explicabilidade nesta pasta para atualizar.</p>')
        statuses=[]
    if statuses:
        counters=Counter(r['Status'] for r in statuses)
        pieces.append('<p>'+html.escape(str(dict(counters)))+'</p>')
        pieces.append('<p>Até duas imagens de cada classe, escolhidas aleatoriamente com semente fixa, sem seleção visual por qualidade dos mapas. '
            'Amostra exploratória do checkpoint escolhido pela validação. Todos os métodos explicam a classe PSOR, inclusive nas imagens NonPsor. '
            'Grad-CAM mostra contribuição positiva ao logit; LIME aproxima localmente a saída do modelo; SHAP Partition usa referência preta e orçamento limitado. '
            'Vermelho/azul em LIME e SHAP representam contribuição positiva/negativa. Intensidades são normalizadas por mapa e não têm escala comum entre métodos. '
            'R² local descreve fidelidade do LIME nas perturbações; não mede acurácia diagnóstica. Descrições automáticas não avaliam localização clínica das lesões.</p>')
        pieces.append(html_table(['Imagem','Método','Status','Alvo','R² LIME','Descrição'],
            [[r['ID'],r['Metodo'],r['Status'],r['Classe_explicada'],r.get('R2_local',''),r['Detalhe']] for r in statuses]))
        xai_resources=read_csv(output/'csv'/'recursos_explicabilidade.csv')
        pieces.append(html_table(['Imagem','Etapa','Tempo (s)','Pico RAM (MiB)'],
            [[r['Imagem'],r['Fase'],number(float(r['Tempo_s']),2),number(float(r['RAM_RSS_pico_MiB']),1)] for r in xai_resources]))
        path=output/'png'/'03_explicabilidade.png'
        if path.exists():
            encoded=base64.b64encode(path.read_bytes()).decode()
            pieces.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="Comparação dos métodos de explicabilidade"></figure>')
    else: pieces.append('<p>Explicabilidade ainda não gerada ou desativada. Resultados de treino e teste permanecem disponíveis.</p>')
    pieces.append('<h2>6. Alcance e rastreabilidade</h2><p>A auditoria verifica duplicatas exatas; não identifica pacientes nem recortes diferentes da mesma fotografia. '
        'O agrupamento por paciente/imagem original deve ser garantido no separador. O teste é utilizado para avaliação, não para selecionar arquitetura ou semente. '
        'Resultados caracterizam a partição e o hardware registrados. Não constituem validação clínica ou medição em smartphones.</p>')
    pieces.append('<p>Arquivos: <b>csv/</b> tabelas completas; <b>png/</b> figuras separadas; <b>checkpoints/</b> pesos; '
        '<b>dados/</b> resultados intermediários e mapas numéricos; <b>logs/</b> execução e erros. '
        'Configuração, versões, contagens e assinatura da base acompanham o relatório.</p>')
    sessions=read(output/'sessoes.json') if (output/'sessoes.json').exists() else []
    if sessions:
        pieces.append(html_table(['Sessão','Modo','Tempo registrado (s)'],[[r['inicio'],r['modo'],number(r['tempo_s'],1)] for r in sessions]))
        pieces.append('<p>Tempos de sessão incluem trabalho do controlador até a montagem final do relatório; não incluem o tempo do navegador nem somam períodos entre retomadas.</p>')
    pieces.append('<details><summary>Configuração e ambiente</summary><pre>'+html.escape(json.dumps(config,ensure_ascii=False,indent=2))+'</pre></details>')
    style='body{font:16px/1.55 Arial,sans-serif;color:#213547;background:#f5f7fa;margin:0}main{max-width:1400px;margin:auto;padding:36px;background:white}h1{color:#123c53}h2{margin-top:36px;border-bottom:2px solid #dae6ec;padding-bottom:8px}.intro{font-size:20px}.card{background:#eaf4f8;padding:18px;border-left:5px solid #0072b2}.notice{background:#fff0d8;padding:15px}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;border-bottom:1px solid #dce2e8;text-align:left}th{background:#eaf0f5}tr:nth-child(even){background:#f8fafc}figure{margin:30px 0}img{max-width:100%;height:auto}figcaption{font-size:13px;color:#536675}pre{white-space:pre-wrap;font-size:12px}@media print{main{padding:0}h2{break-after:avoid}figure{break-inside:avoid}.table{overflow:visible}body{background:white}}'
    document='<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Relatório da comparação de modelos</title><style>'+style+'</style><main>'+''.join(pieces)+'</main></html>'
    (output/'relatorio_final.html').write_text(document,encoding='utf-8')
    text=[f'RELATÓRIO — {len(rows)}/{expected} execuções concluídas',f'Dispositivo: {config["device"]}',
          'Seleção por validação: '+(DISPLAY[choice['Modelo']] if choice else 'indisponível'),
          'Métricas: média ± desvio-padrão entre sementes; partição fixa.']
    for s in summaries:
        text.append(f"{DISPLAY[s['Modelo']]} | F1 teste {mean_sd(s,'Test_F1')} | AUC {mean_sd(s,'Test_AUC')} | "
            f"latência mediana {mean_sd(s,'Forward_mediana_ms',2)} ms | RAM pico inferência {mean_sd(s,'Inferencia_RAM_pico_MiB',1)} MiB")
    text.extend(['Detalhes, gráficos e definições: relatorio_final.html.',
        'RSS inclui o processo Python; energia elétrica não medida. Auditoria exata não verifica parentesco de imagens/pacientes.'])
    (output/'resumo_final.txt').write_text('\n'.join(text),encoding='utf-8')
    return choice


def protocol(base,signature):
    return {'versao':VERSION,'base':str(Path(base).resolve()),'assinatura_dados':signature,
        'modelos':MODELS,'sementes':SEEDS,'resolucao':SIZE,'batch':BATCH,'epocas':EPOCHS,'paciencia':PATIENCE,
        'lr':LR,'limiar':THRESHOLD,'media':MEAN,'desvio':STD,'threads':THREADS,'device':str(DEVICE),
        'benchmark_repeticoes':BENCH_REPEATS,'warmup':WARMUP,'amostragem_RAM_s':MEM_INTERVAL,
        'xai_por_classe':XAI_PER_CLASS,'lime_amostras':LIME_SAMPLES,'shap_avaliacoes':SHAP_EVALS,
        'xai_classe_alvo':'PSOR','codigo_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':platform.python_version(),'torch':torch.__version__,
        'torchvision':torchvision.__version__,'numpy':np.__version__,'psutil':psutil.__version__,
        'sistema':platform.platform(),'cpu':platform.processor(),'nucleos_logicos':os.cpu_count(),
        'RAM_total_GiB':psutil.virtual_memory().total/1024**3,
        'gpu':torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else None}


def check_existing(output,base):
    config=read(output/'configuracao.json')
    if config.get('versao')!=VERSION:
        raise ValueError('Retomada exige pasta produzida por esta versão V5. Não use os checkpoints da V4.')
    signature,_=audit(datasets(base))
    current=protocol(base,signature)
    critical=['assinatura_dados','modelos','sementes','resolucao','batch','epocas','paciencia','lr','limiar',
              'media','desvio','threads','device','torch','torchvision','python','benchmark_repeticoes','warmup','codigo_sha256']
    differences=[k for k in critical if current[k]!=config[k]]
    if differences:raise ValueError('Dados/configuração divergentes: '+', '.join(differences))
    return config


def child(output,mode,name=None,seed=None):
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',mode,'--run',str(output)]
    if name:command+=['--model',name,'--seed',str(seed)]
    tag=f'{mode}_{name}_s{seed}' if name else mode
    log=output/'logs'/f'{tag}.txt'
    start=time.perf_counter()
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
    env['PYTHONHASHSEED']=str(seed if seed is not None else XAI_SEED)
    with log.open('w',encoding='utf-8') as stream:
        with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',errors='replace',env=env) as process:
            try:
                for line in process.stdout:
                    print(line,end='',flush=True);stream.write(line);stream.flush()
                code=process.wait()
            except KeyboardInterrupt:
                process.terminate();process.wait();raise
    record={'Modo':mode,'Modelo':name or '', 'Semente':seed if seed is not None else '',
            'Status':'OK' if code==0 else 'ERRO','Tempo_subprocesso_s':time.perf_counter()-start,'Log':str(log)}
    statuses=read_csv(output/'csv'/'status_execucoes.csv');statuses.append(record)
    write_csv(output/'csv'/'status_execucoes.csv',statuses)
    return code==0


def preflight(base,skip_xai):
    data=datasets(base);_,counts=audit(data)
    print('Distribuição:',counts,flush=True)
    tools=optional_tools(required=not skip_xai)
    tensor=data['Validacao'][0][0][None].to(DEVICE)
    for name in MODELS:
        model=architecture(name,pretrained=False).eval()
        for target in [0,1]:
            cam,_=gradcam(model,name,tensor,target)
            if cam.shape!=(SIZE,SIZE):raise ValueError(f'Mapa inválido: {name}')
        print(f'Forward/Grad-CAM: {name} OK',flush=True)
        del model;gc.collect()
    if not skip_xai:
        model=architecture('mobilenet_v3_small',pretrained=False).eval()
        rgb=(tensor[0].cpu().permute(1,2,0).numpy()*np.asarray(STD)+np.asarray(MEAN)).clip(0,1).astype(np.float32)
        predict=rgb_predictor(model)
        lime_map(tools,rgb,predict,samples=32)
        shap_map(tools,rgb,predict,evaluations=32)
        print('LIME e SHAP: teste reduzido de execução OK (pesos aleatórios; não constitui resultado científico).')
    with Meter('Teste_monitor_memoria') as meter:
        probe=bytearray(4*1024**2)
        time.sleep(.12)
        del probe
    print('Monitor de memória OK:',meter.result['RAM_RSS_pico_MiB'],'MiB')
    print('Pré-verificação concluída. Nenhum treinamento foi executado.')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base',type=Path,default=None)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--verificar',action='store_true')
    action.add_argument('--retomar',type=Path)
    action.add_argument('--so-relatorio',type=Path)
    action.add_argument('--so-explicabilidade',type=Path)
    parser.add_argument('--sem-explicabilidade',action='store_true',help='Pula LIME/SHAP/Grad-CAM nesta execução.')
    parser.add_argument('--worker',choices=['train','eval','xai'],help=argparse.SUPPRESS)
    parser.add_argument('--run',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--model',choices=MODELS,help=argparse.SUPPRESS)
    parser.add_argument('--seed',type=int,default=42,help=argparse.SUPPRESS)
    args=parser.parse_args()
    configure(args.seed)
    if args.worker:
        if args.worker=='train':train_worker(args.run,args.model,args.seed)
        elif args.worker=='eval':eval_worker(args.run,args.model,args.seed)
        else:xai_worker(args.run)
        return
    if args.so_relatorio:
        output=args.so_relatorio.resolve()
        if read(output/'configuracao.json').get('versao')!=VERSION:raise ValueError('Pasta não é da V5.')
        report(output)
        print('Relatório atualizado:',output/'relatorio_final.html');return
    if args.verificar:
        preflight(args.base or BASE,args.sem_explicabilidade);return
    if not args.sem_explicabilidade:optional_tools()
    started=time.perf_counter();when=datetime.now().isoformat(timespec='seconds')
    existing=args.retomar or args.so_explicabilidade
    if existing:
        output=existing.resolve()
        config=read(output/'configuracao.json')
        base=args.base or Path(config['base'])
        check_existing(output,base)
    else:
        base=args.base or BASE
        output=base/'Resultados_Rinha_V5'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        for folder in ['csv','png','checkpoints','dados','logs']:(output/folder).mkdir(parents=True,exist_ok=True)
        signature,counts=audit(datasets(base),output)
        dump(output/'configuracao.json',protocol(base,signature))
        # Cópia exata do código usado para rastreabilidade da execução local.
        (output/'codigo_executado.py').write_bytes(Path(__file__).read_bytes())
        print('Distribuição:',counts)
    print('Pasta da execução:',output,flush=True)
    if not args.so_explicabilidade:
        for name in MODELS:
            for seed in SEEDS:
                _,checkpoint,trained,evaluated=paths(output,name,seed)
                if evaluated.exists() and checkpoint.exists():
                    print(f'Reutilizando execução concluída: {name}, semente {seed}');continue
                if not trained.exists() or not checkpoint.exists():
                    if not child(output,'train',name,seed):continue
                child(output,'eval',name,seed)
    rows,summaries=collect(output)
    choice=select(output,rows,summaries)
    if choice and not args.sem_explicabilidade:
        # Uma nova solicitação gera novamente XAI para o checkpoint selecionado,
        # mas nunca reinicia os treinamentos concluídos.
        child(output,'xai')
    sessions=read(output/'sessoes.json') if (output/'sessoes.json').exists() else []
    sessions.append({'inicio':when,'modo':'so_explica' if args.so_explicabilidade else ('retomada' if args.retomar else 'completa'),
                     'tempo_s':time.perf_counter()-started})
    dump(output/'sessoes.json',sessions)
    report(output)
    print('\nRelatório:',output/'relatorio_final.html')
    print('PNGs:',output/'png')
    print('CSVs:',output/'csv')
    print(f'Execuções completas: {len(rows)}/{len(MODELS)*len(SEEDS)}. Consulte status e logs para pendências.')


if __name__=='__main__':
    main()
