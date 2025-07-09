# Starting structure for a Zig-based tensor library
.
├── build.zig
├── build.zig.zon
└── src/
    ├── lib.zig         // O ficheiro principal que exporta a API pública da biblioteca.
    ├── tensor.zig      // A definição da struct principal `Tensor`.
    |
    ├── ops/            // Diretório para todas as operações de tensor.
    │   ├── arithmetic.zig
    │   ├── linalg.zig    // Álgebra linear, ex: matmul.
    │   └── creation.zig  // Funções como `zeros`, `ones`, `rand`.
    |
    ├── autograd.zig    // Lógica para o grafo computacional e backpropagation.
    |
    └── nn/             // Módulos de alto nível para redes neuronais.
        ├── module.zig
        ├── layer.zig
        └── activation.zig

# Future Enhancements
├── build.zig                 // Script de compilação avançado com flags para backends (CPU, CUDA)
├── build.zig.zon
├── examples/                 // Exemplos de uso da biblioteca
├── tests/                    // Testes de unidade e de integração
├── docs/                     // Documentação
├── bindings/
│   └── python/               // Código específico para os bindings de Python (C-API, setup.py)
│
└── src/
    ├── lib.zig               // API pública principal, exporta os símbolos mais importantes
    │
    ├── core/                 // Componentes fundamentais e de baixo nível
    │   ├── tensor.zig        // Definição da struct Tensor
    │   ├── storage.zig       // Abstração sobre a memória bruta (o buffer de dados)
    │   ├── device.zig        // Abstração para CPU, GPU (ex: `Device.Cpu`, `Device.Cuda(0)`)
    │   ├── dtype.zig         // Enum para tipos de dados (f32, f16, int8, etc.)
    │   └── shape.zig         // Utilitários para manipulação de forma (shape) e broadcasting
    │
    ├── backend/              // Abstração do hardware
    │   ├── backend.zig       // Interface genérica que todos os backends devem implementar
    │   ├── cpu/              // Implementação do backend para CPU
    │   │   └── kernels/      // Código otimizado (ex: com SIMD) para operações na CPU
    │   └── cuda/             // Implementação do backend para CUDA
    │       ├── cuda.zig      // Bindings para a API do driver CUDA
    │       ├── cublas.zig    // Bindings para a biblioteca cuBLAS (álgebra linear)
    │       └── kernels/      // Ficheiros .cu com kernels CUDA para operações personalizadas
    │
    ├── ops/                    // Definição das operações (a API lógica)
    │   ├── op.zig              // Uma interface/struct base para uma operação genérica
    │   ├── arithmetic.zig    // add, sub, mul, div (despacham para o backend correto)
    │   ├── linalg.zig        // matmul, dot (despacham para o backend correto)
    │   └── creation.zig      // zeros, ones, randn (despacham para o backend correto)
    │
    ├── autograd/               // Motor de diferenciação automática
    │   ├── engine.zig          // O motor que executa a backpropagation
    │   ├── function.zig        // A base para cada função diferenciável (com `forward` e `backward`)
    │   └── graph.zig           // Estruturas de dados para representar o grafo computacional
    │
    ├── nn/                     // A biblioteca de redes neuronais de alto nível
    │   ├── module.zig          // A struct base `Module`, similar à do PyTorch
    │   ├── functional.zig      // Funções puras (sem estado) como F.relu, F.softmax
    │   ├── init.zig            // Funções de inicialização de pesos (Xavier, Kaiming, etc.)
    │   ├── loss.zig            // Funções de perda (CrossEntropy, MSELoss)
    │   ├── optimizer.zig       // Otimizadores (SGD, Adam, RMSprop)
    │   └── layers/             // Diretório para as diferentes camadas
    │       ├── linear.zig
    │       ├── conv.zig
    │       ├── recurrent.zig   // RNN, LSTM, GRU
    │       └── attention.zig   // Multi-Head Attention para Transformers
    │
    ├── data/                   // Utilitários para carregamento e manipulação de dados
    │   ├── dataset.zig         // Abstração para um conjunto de dados
    │   ├── dataloader.zig      // Carregador de dados (batching, shuffling, multi-processamento)
    │   └── transforms.zig      // Transformações de dados (normalização, data augmentation)
    │
    └── serialize/              // Funcionalidades para guardar e carregar modelos
        ├── save_load.zig       // API principal para `save()` e `load()`
        └── formats/            // Suporte para diferentes formatos (ex: ONNX, SafeTensors)
