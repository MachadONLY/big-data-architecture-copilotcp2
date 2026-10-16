# GPU / CUDA

## Primeiro comando

Antes de instalar ou alterar qualquer versão:

```powershell
nvidia-smi
```

Registre:

- modelo da GPU;
- VRAM;
- driver;
- campo `CUDA Version` mostrado pelo driver.

## Atenção: duas versões CUDA diferentes

O campo `CUDA Version` do `nvidia-smi` indica a **compatibilidade máxima de CUDA exposta pelo driver**.

Ele não prova qual CUDA Toolkit (`nvcc`) está instalado.

Para Toolkit:

```powershell
nvcc --version
```

Se `nvcc` não existir, isso não significa automaticamente que PyTorch CUDA não possa funcionar: wheels do PyTorch podem trazer runtimes CUDA próprios.

## Python e sistema

```powershell
python --version
python -m pip --version
git --version
```

## Instalação de PyTorch

Não há um comando CUDA hardcoded neste repositório porque a wheel correta depende da GPU/driver e das versões disponíveis no momento da instalação.

Use o seletor oficial:

https://pytorch.org/get-started/locally/

Selecione:

- Windows
- Pip
- Python
- CUDA compatível com sua máquina

Depois valide:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Somente depois instale:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## Checagem do projeto

```powershell
python src/gpu_check.py --require-cuda --output assets/screenshots/gpu_environment.json
python src/verify_stack.py --require-cuda --output assets/screenshots/stack_environment.json
```

Ele mostra:

- `torch.cuda.is_available()`;
- versão PyTorch;
- runtime CUDA usado pelo PyTorch;
- GPU;
- VRAM;
- compute capability;
- suporte BF16;
- device atual;
- `nvidia-smi`;
- `nvcc --version`, quando disponível.

`verify_stack.py` valida as versões fixadas, o contrato do `SFTConfig`, a
configuração 4-bit e uma quantização/dequantização NF4 real do bitsandbytes na GPU.
Se esse smoke test falhar, não inicie os experimentos.

## Evidência visual para a apresentação

Durante o treinamento:

Terminal 1:

```powershell
python src/train.py --config configs/experiment_01.yaml
```

Terminal 2:

```powershell
nvidia-smi -l 2
```

Tire um screenshot no VSCode contendo:

1. editor/arquivo `train.py`;
2. terminal mostrando epoch/step/loss;
3. `nvidia-smi`;
4. processo `python.exe`;
5. VRAM ocupada;
6. GPU utilization.

Isso é evidência mais forte do que apenas mostrar um plugin do VSCode.

## Plugins sugeridos pelo professor

- GPU Environment
- GPU Monitor

Eles podem ser instalados como apoio visual, mas não substituem CUDA/PyTorch/`nvidia-smi`.

## bitsandbytes

A documentação atual do bitsandbytes oferece builds NVIDIA para Windows x86-64 em faixas modernas de CUDA. Ainda assim, a compatibilidade deve ser validada na GPU específica antes da entrega.

Referência:

https://huggingface.co/docs/bitsandbytes/en/installation
