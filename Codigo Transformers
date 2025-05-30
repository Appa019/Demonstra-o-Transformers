# Simulação Completa de Attention em Transformers
# Baseado nos vídeos DL5 e DL6 sobre Transformers e Attention

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
import pandas as pd
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Configuração para melhor visualização
plt.style.use('default')
sns.set_palette("husl")

print("🤖 SIMULAÇÃO COMPLETA DE ATTENTION EM TRANSFORMERS")
print("=" * 60)

# =============================================================================
# PARTE 1: FUNDAMENTOS - EMBEDDINGS E POSICIONAL ENCODING
# =============================================================================

class TransformerSimulator:
    def __init__(self, d_model=64, seq_length=8, vocab_size=1000):
        self.d_model = d_model
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Inicializar pesos aleatórios (simplificado)
        np.random.seed(42)
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        
    def create_positional_encoding(self):
        """Cria encoding posicional sinusoidal"""
        pe = np.zeros((self.seq_length, self.d_model))
        
        for pos in range(self.seq_length):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** (i / self.d_model)))
                if i + 1 < self.d_model:
                    pe[pos, i + 1] = np.cos(pos / (10000 ** (i / self.d_model)))
        
        return pe
    
    def create_sample_embeddings(self):
        """Cria embeddings de exemplo para uma frase"""
        # Simular embeddings para tokens: ["O", "gato", "subiu", "no", "telhado", "ontem", "à", "noite"]
        embeddings = np.random.randn(self.seq_length, self.d_model) * 0.5
        
        # Adicionar alguma estrutura semântica simulada
        embeddings[0] *= 0.8  # "O" - artigo
        embeddings[1] += 0.3  # "gato" - substantivo
        embeddings[2] += 0.5  # "subiu" - verbo
        embeddings[3] *= 0.7  # "no" - preposição
        embeddings[4] += 0.4  # "telhado" - substantivo
        embeddings[5] += 0.2  # "ontem" - advérbio temporal
        embeddings[6] *= 0.6  # "à" - preposição
        embeddings[7] += 0.1  # "noite" - substantivo
        
        return embeddings
    
    def visualize_embeddings_and_positional(self):
        """Visualiza embeddings e encoding posicional"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Embeddings originais
        embeddings = self.create_sample_embeddings()
        pos_encoding = self.create_positional_encoding()
        
        # 1. Embeddings de tokens
        im1 = axes[0,0].imshow(embeddings.T, cmap='RdBu', aspect='auto')
        axes[0,0].set_title('1. Token Embeddings')
        axes[0,0].set_xlabel('Posição na Sequência')
        axes[0,0].set_ylabel('Dimensões do Embedding')
        axes[0,0].set_xticks(range(8))
        axes[0,0].set_xticklabels(['O', 'gato', 'subiu', 'no', 'telhado', 'ontem', 'à', 'noite'])
        plt.colorbar(im1, ax=axes[0,0])
        
        # 2. Positional Encoding
        im2 = axes[0,1].imshow(pos_encoding.T, cmap='viridis', aspect='auto')
        axes[0,1].set_title('2. Positional Encoding')
        axes[0,1].set_xlabel('Posição na Sequência')
        axes[0,1].set_ylabel('Dimensões do Encoding')
        plt.colorbar(im2, ax=axes[0,1])
        
        # 3. Embeddings finais (token + posicional)
        final_embeddings = embeddings + pos_encoding
        im3 = axes[1,0].imshow(final_embeddings.T, cmap='RdBu', aspect='auto')
        axes[1,0].set_title('3. Embeddings Finais (Token + Posicional)')
        axes[1,0].set_xlabel('Posição na Sequência')
        axes[1,0].set_ylabel('Dimensões')
        axes[1,0].set_xticks(range(8))
        axes[1,0].set_xticklabels(['O', 'gato', 'subiu', 'no', 'telhado', 'ontem', 'à', 'noite'])
        plt.colorbar(im3, ax=axes[1,0])
        
        # 4. Padrões do Positional Encoding
        axes[1,1].plot(pos_encoding[:, :10])
        axes[1,1].set_title('4. Padrões Sinusoidais do Positional Encoding')
        axes[1,1].set_xlabel('Posição')
        axes[1,1].set_ylabel('Valor')
        axes[1,1].legend([f'Dim {i}' for i in range(10)], bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        return final_embeddings

# =============================================================================
# PARTE 2: MECANISMO DE SELF-ATTENTION PASSO A PASSO
# =============================================================================

    def compute_attention_step_by_step(self, X):
        """Computa self-attention passo a passo com visualizações"""
        print("\n" + "="*50)
        print("PASSO A PASSO: SELF-ATTENTION MECHANISM")
        print("="*50)
        
        # Passo 1: Criar Q, K, V
        print("\n🔍 PASSO 1: Criando Query, Key e Value matrices")
        Q = X @ self.W_q  # Queries
        K = X @ self.W_k  # Keys  
        V = X @ self.W_v  # Values
        
        print(f"Forma de X (input): {X.shape}")
        print(f"Forma de Q (queries): {Q.shape}")
        print(f"Forma de K (keys): {K.shape}")
        print(f"Forma de V (values): {V.shape}")
        
        # Visualizar Q, K, V
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        tokens = ['O', 'gato', 'subiu', 'no', 'telhado', 'ontem', 'à', 'noite']
        
        im1 = axes[0].imshow(Q.T, cmap='Reds', aspect='auto')
        axes[0].set_title('Query Matrix (Q)')
        axes[0].set_xlabel('Tokens')
        axes[0].set_ylabel('Dimensões')
        axes[0].set_xticks(range(8))
        axes[0].set_xticklabels(tokens, rotation=45)
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(K.T, cmap='Greens', aspect='auto')
        axes[1].set_title('Key Matrix (K)')
        axes[1].set_xlabel('Tokens')
        axes[1].set_ylabel('Dimensões')
        axes[1].set_xticks(range(8))
        axes[1].set_xticklabels(tokens, rotation=45)
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(V.T, cmap='Blues', aspect='auto')
        axes[2].set_title('Value Matrix (V)')
        axes[2].set_xlabel('Tokens')
        axes[2].set_ylabel('Dimensões')
        axes[2].set_xticks(range(8))
        axes[2].set_xticklabels(tokens, rotation=45)
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.show()
        
        # Passo 2: Calcular Attention Scores
        print("\n🧮 PASSO 2: Calculando Attention Scores (Q @ K^T)")
        attention_scores = Q @ K.T
        
        # Escalar por sqrt(d_k) 
        scaled_scores = attention_scores / np.sqrt(self.d_model)
        
        print(f"Forma dos Attention Scores: {attention_scores.shape}")
        print(f"Escala aplicada: 1/√{self.d_model} = {1/np.sqrt(self.d_model):.3f}")
        
        # Passo 3: Aplicar Softmax
        print("\n📊 PASSO 3: Aplicando Softmax para obter Attention Weights")
        attention_weights = softmax(scaled_scores, axis=-1)
        
        # Visualizar scores e weights
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Attention Scores (antes do softmax)
        im1 = axes[0].imshow(scaled_scores, cmap='RdYlBu', aspect='auto')
        axes[0].set_title('Attention Scores (Escalados)')
        axes[0].set_xlabel('Key Positions')
        axes[0].set_ylabel('Query Positions')
        axes[0].set_xticks(range(8))
        axes[0].set_yticks(range(8))
        axes[0].set_xticklabels(tokens)
        axes[0].set_yticklabels(tokens)
        plt.colorbar(im1, ax=axes[0])
        
        # Attention Weights (depois do softmax)
        im2 = axes[1].imshow(attention_weights, cmap='YlOrRd', aspect='auto')
        axes[1].set_title('Attention Weights (após Softmax)')
        axes[1].set_xlabel('Key Positions')
        axes[1].set_ylabel('Query Positions')
        axes[1].set_xticks(range(8))
        axes[1].set_yticks(range(8))
        axes[1].set_xticklabels(tokens)
        axes[1].set_yticklabels(tokens)
        
        # Adicionar valores dos pesos na visualização
        for i in range(8):
            for j in range(8):
                text = axes[1].text(j, i, f'{attention_weights[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im2, ax=axes[1])
        plt.tight_layout()
        plt.show()
        
        # Verificar se as linhas somam 1 (propriedade do softmax)
        print(f"Verificação Softmax - Soma de cada linha: {attention_weights.sum(axis=1)}")
        
        # Passo 4: Aplicar pesos aos Values
        print("\n🎯 PASSO 4: Computando Output (Attention × Values)")
        output = attention_weights @ V
        
        print(f"Forma do Output: {output.shape}")
        
        # Visualizar o output
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        im1 = axes[0].imshow(V.T, cmap='Blues', aspect='auto')
        axes[0].set_title('Values Matrix (V)')
        axes[0].set_xlabel('Tokens')
        axes[0].set_ylabel('Dimensões')
        axes[0].set_xticks(range(8))
        axes[0].set_xticklabels(tokens)
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(output.T, cmap='Purples', aspect='auto')
        axes[1].set_title('Attention Output')
        axes[1].set_xlabel('Tokens')
        axes[1].set_ylabel('Dimensões')
        axes[1].set_xticks(range(8))
        axes[1].set_xticklabels(tokens)
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
        
        return Q, K, V, attention_weights, output

# =============================================================================
# PARTE 3: ANÁLISE DETALHADA DOS PADRÕES DE ATENÇÃO
# =============================================================================

    def analyze_attention_patterns(self, attention_weights):
        """Analisa padrões específicos de atenção"""
        print("\n" + "="*50)
        print("ANÁLISE DE PADRÕES DE ATENÇÃO")
        print("="*50)
        
        tokens = ['O', 'gato', 'subiu', 'no', 'telhado', 'ontem', 'à', 'noite']
        
        # 1. Atenção por token específico
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Análise para "gato" (posição 1)
        gato_attention = attention_weights[1, :]
        axes[0,0].bar(range(8), gato_attention, color='skyblue', alpha=0.7)
        axes[0,0].set_title('Atenção de "gato" para outros tokens')
        axes[0,0].set_xlabel('Tokens')
        axes[0,0].set_ylabel('Peso de Atenção')
        axes[0,0].set_xticks(range(8))
        axes[0,0].set_xticklabels(tokens, rotation=45)
        
        # Adicionar valores nas barras
        for i, v in enumerate(gato_attention):
            axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Análise para "subiu" (posição 2)
        subiu_attention = attention_weights[2, :]
        axes[0,1].bar(range(8), subiu_attention, color='lightcoral', alpha=0.7)
        axes[0,1].set_title('Atenção de "subiu" para outros tokens')
        axes[0,1].set_xlabel('Tokens')
        axes[0,1].set_ylabel('Peso de Atenção')
        axes[0,1].set_xticks(range(8))
        axes[0,1].set_xticklabels(tokens, rotation=45)
        
        for i, v in enumerate(subiu_attention):
            axes[0,1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Mapa de calor com anotações
        sns.heatmap(attention_weights, annot=True, fmt='.3f', 
                   xticklabels=tokens, yticklabels=tokens,
                   cmap='YlOrRd', ax=axes[1,0])
        axes[1,0].set_title('Matriz de Atenção Completa')
        axes[1,0].set_xlabel('Attending to (Keys)')
        axes[1,0].set_ylabel('Attending from (Queries)')
        
        # Distribuição dos pesos de atenção
        axes[1,1].hist(attention_weights.flatten(), bins=20, alpha=0.7, color='green')
        axes[1,1].set_title('Distribuição dos Pesos de Atenção')
        axes[1,1].set_xlabel('Valor do Peso')
        axes[1,1].set_ylabel('Frequência')
        axes[1,1].axvline(attention_weights.mean(), color='red', linestyle='--', 
                         label=f'Média: {attention_weights.mean():.3f}')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # 2. Análise de relações sintáticas/semânticas
        print("\n🔍 ANÁLISE DE RELAÇÕES DETECTADAS:")
        print("-" * 40)
        
        for i, token_query in enumerate(tokens):
            top_attended = np.argsort(attention_weights[i])[::-1][:3]
            print(f"'{token_query}' presta mais atenção em:")
            for j, idx in enumerate(top_attended):
                if idx != i:  # Não incluir self-attention
                    print(f"  {j+1}. '{tokens[idx]}' (peso: {attention_weights[i, idx]:.3f})")
        
        return attention_weights

# =============================================================================
# PARTE 4: MULTI-HEAD ATTENTION
# =============================================================================

class MultiHeadAttention:
    def __init__(self, d_model=64, num_heads=8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        np.random.seed(42)
        # Pesos para cada cabeça
        self.W_q_heads = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(num_heads)]
        self.W_k_heads = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(num_heads)]
        self.W_v_heads = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(num_heads)]
        
        # Projeção final
        self.W_o = np.random.randn(d_model, d_model) * 0.1
    
    def single_head_attention(self, X, W_q, W_k, W_v):
        """Computa atenção para uma única cabeça"""
        Q = X @ W_q
        K = X @ W_k
        V = X @ W_v
        
        attention_scores = Q @ K.T / np.sqrt(self.d_k)
        attention_weights = softmax(attention_scores, axis=-1)
        output = attention_weights @ V
        
        return output, attention_weights
    
    def compute_multi_head_attention(self, X):
        """Computa multi-head attention"""
        print("\n" + "="*50)
        print("MULTI-HEAD ATTENTION")
        print("="*50)
        
        head_outputs = []
        head_attentions = []
        
        # Computar cada cabeça
        for i in range(self.num_heads):
            output, attention = self.single_head_attention(
                X, self.W_q_heads[i], self.W_k_heads[i], self.W_v_heads[i]
            )
            head_outputs.append(output)
            head_attentions.append(attention)
        
        # Concatenar outputs das cabeças
        concatenated = np.concatenate(head_outputs, axis=-1)
        
        # Projeção final
        final_output = concatenated @ self.W_o
        
        print(f"Número de cabeças: {self.num_heads}")
        print(f"Dimensão por cabeça (d_k): {self.d_k}")
        print(f"Forma do output concatenado: {concatenated.shape}")
        print(f"Forma do output final: {final_output.shape}")
        
        # Visualizar diferentes cabeças
        self.visualize_multi_head_patterns(head_attentions)
        
        return final_output, head_attentions
    
    def visualize_multi_head_patterns(self, head_attentions):
        """Visualiza padrões de atenção de diferentes cabeças"""
        tokens = ['O', 'gato', 'subiu', 'no', 'telhado', 'ontem', 'à', 'noite']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i in range(self.num_heads):
            im = axes[i].imshow(head_attentions[i], cmap='YlOrRd', aspect='auto')
            axes[i].set_title(f'Cabeça {i+1}')
            axes[i].set_xticks(range(8))
            axes[i].set_yticks(range(8))
            axes[i].set_xticklabels(tokens, rotation=45, fontsize=8)
            axes[i].set_yticklabels(tokens, fontsize=8)
            plt.colorbar(im, ax=axes[i])
        
        plt.suptitle('Padrões de Atenção por Cabeça', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Análise de especialização das cabeças
        print("\n🎯 ESPECIALIZAÇÃO DAS CABEÇAS:")
        print("-" * 40)
        
        for i in range(self.num_heads):
            attention = head_attentions[i]
            
            # Calcular se a cabeça foca em posições próximas ou distantes
            local_attention = 0
            distant_attention = 0
            
            for row in range(len(attention)):
                for col in range(len(attention[row])):
                    distance = abs(row - col)
                    if distance <= 1:
                        local_attention += attention[row, col]
                    elif distance >= 3:
                        distant_attention += attention[row, col]
            
            if local_attention > distant_attention:
                pattern = "LOCAL (posições próximas)"
            else:
                pattern = "GLOBAL (posições distantes)"
            
            print(f"Cabeça {i+1}: {pattern}")
            print(f"  - Atenção local: {local_attention:.3f}")
            print(f"  - Atenção distante: {distant_attention:.3f}")

# =============================================================================
# PARTE 5: TRANSFORMER BLOCK COMPLETO
# =============================================================================

class TransformerBlock:
    def __init__(self, d_model=64, num_heads=8, d_ff=256):
        self.d_model = d_model
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed Forward Network
        np.random.seed(42)
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
    
    def layer_norm(self, x, epsilon=1e-6):
        """Layer Normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + epsilon)
    
    def feed_forward(self, x):
        """Feed Forward Network com ReLU"""
        hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        output = hidden @ self.W2 + self.b2
        return output
    
    def forward(self, x):
        """Forward pass completo do transformer block"""
        print("\n" + "="*50)
        print("TRANSFORMER BLOCK COMPLETO")
        print("="*50)
        
        # 1. Multi-Head Attention + Residual Connection + Layer Norm
        print("\n🔄 Etapa 1: Multi-Head Attention")
        attn_output, head_attentions = self.multi_head_attention.compute_multi_head_attention(x)
        
        # Residual connection
        x_after_attn = x + attn_output
        print(f"Forma após residual connection: {x_after_attn.shape}")
        
        # Layer normalization
        x_norm1 = self.layer_norm(x_after_attn)
        print(f"Forma após layer norm 1: {x_norm1.shape}")
        
        # 2. Feed Forward + Residual Connection + Layer Norm
        print("\n🧠 Etapa 2: Feed Forward Network")
        ff_output = self.feed_forward(x_norm1)
        print(f"Forma após feed forward: {ff_output.shape}")
        
        # Residual connection
        x_after_ff = x_norm1 + ff_output
        
        # Layer normalization
        final_output = self.layer_norm(x_after_ff)
        print(f"Forma do output final: {final_output.shape}")
        
        # Visualizar transformações
        self.visualize_transformer_stages(x, x_after_attn, x_norm1, ff_output, final_output)
        
        return final_output, head_attentions
    
    def visualize_transformer_stages(self, x_input, x_attn, x_norm1, x_ff, x_final):
        """Visualiza as diferentes etapas do transformer block"""
        tokens = ['O', 'gato', 'subiu', 'no', 'telhado', 'ontem', 'à', 'noite']
        
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        stages = [
            (x_input, 'Input'),
            (x_attn, 'Após Attention'),
            (x_norm1, 'Após LayerNorm 1'),
            (x_ff, 'Após Feed Forward'),
            (x_final, 'Output Final')
        ]
        
        for i, (data, title) in enumerate(stages):
            im = axes[i].imshow(data.T, cmap='RdBu', aspect='auto')
            axes[i].set_title(title)
            axes[i].set_xlabel('Tokens')
            axes[i].set_ylabel('Dimensões')
            axes[i].set_xticks(range(8))
            axes[i].set_xticklabels(tokens, rotation=45)
            plt.colorbar(im, ax=axes[i])
        
        plt.suptitle('Transformações no Transformer Block', fontsize=16)
        plt.tight_layout()
        plt.show()

# =============================================================================
# PARTE 6: SIMULAÇÃO COMPLETA E ANÁLISE
# =============================================================================

def run_complete_simulation():
    """Executa simulação completa com análises"""
    print("🚀 INICIANDO SIMULAÇÃO COMPLETA DO TRANSFORMER")
    print("="*60)
    
    # Inicializar simulador
    sim = TransformerSimulator(d_model=64, seq_length=8)
    
    # 1. Embeddings e Positional Encoding
    print("\n📊 ETAPA 1: EMBEDDINGS E POSITIONAL ENCODING")
    embeddings = sim.visualize_embeddings_and_positional()
    
    # 2. Self-Attention passo a passo
    print("\n🔍 ETAPA 2: SELF-ATTENTION MECHANISM")
    Q, K, V, attention_weights, attention_output = sim.compute_attention_step_by_step(embeddings)
    
    # 3. Análise de padrões
    print("\n📈 ETAPA 3: ANÁLISE DE PADRÕES")
    sim.analyze_attention_patterns(attention_weights)
    
    # 4. Multi-Head Attention
    print("\n🎯 ETAPA 4: MULTI-HEAD ATTENTION")
    mha = MultiHeadAttention(d_model=64, num_heads=8)
    mha_output, head_attentions = mha.compute_multi_head_attention(embeddings)
    
    # 5. Transformer Block Completo
    print("\n🏗️ ETAPA 5: TRANSFORMER BLOCK COMPLETO")
    transformer_block = TransformerBlock(d_model=64, num_heads=8, d_ff=256)
    final_output, _ = transformer_block.forward(embeddings)
    
    # 6. Comparação Final
    print("\n📊 COMPARAÇÃO FINAL DOS OUTPUTS")
    print("="*40)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    tokens = ['O', 'gato', 'subiu', 'no', 'telhado', 'ontem', 'à', 'noite']
    
    # Input original
    im1 = axes[0,0].imshow(embeddings.T, cmap='RdBu', aspect='auto')
    axes[0,0].set_title('Input Original')
    axes[0,0].set_xticks(range(8))
    axes[0,0].set_xticklabels(tokens, rotation=45)
    plt.colorbar(im1, ax=axes[0,0])
    
    # Single-head attention output
    im2 = axes[0,1].imshow(attention_output.T, cmap='RdBu', aspect='auto')
    axes[0,1].set_title('Single-Head Attention Output')
    axes[0,1].set_xticks(range(8))
    axes[0,1].set_xticklabels(tokens, rotation=45)
    plt.colorbar(im2, ax=axes[0,1])
    
    # Multi-head attention output
    im3 = axes[1,0].imshow(mha_output.T, cmap='RdBu', aspect='auto')
    axes[1,0].set_title('Multi-Head Attention Output')
    axes[1,0].set_xticks(range(8))
    axes[1,0].set_xticklabels(tokens, rotation=45)
    plt.colorbar(im3, ax=axes[1,0])
    
    # Transformer block final output
    im4 = axes[1,1].imshow(final_output.T, cmap='RdBu', aspect='auto')
    axes[1,1].set_title('Transformer Block Final Output')
    axes[1,1].set_xticks(range(8))
    axes[1,1].set_xticklabels(tokens, rotation=45)
    plt.colorbar(im4, ax=axes[1,1])
    
    plt.suptitle('Evolução das Representações através do Transformer', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Métricas finais
    print("\n📋 MÉTRICAS FINAIS:")
    print("-" * 30)
    print(f"Dimensão do modelo: {sim.d_model}")
    print(f"Comprimento da sequência: {sim.seq_length}")
    print(f"Número de cabeças: 8")
    print(f"Dimensão feed-forward: 256")
    print(f"Variância do input: {np.var(embeddings):.4f}")
    print(f"Variância do output final: {np.var(final_output):.4f}")
    print(f"Norma L2 do input: {np.linalg.norm(embeddings):.4f}")
    print(f"Norma L2 do output: {np.linalg.norm(final_output):.4f}")

# =============================================================================
# PARTE 7: VISUALIZAÇÃO INTERATIVA E INTERPRETABILIDADE
# =============================================================================

def create_attention_interpretation():
    """Cria visualizações interpretáveis dos padrões de atenção"""
    print("\n" + "="*50)
    print("INTERPRETABILIDADE DOS PADRÕES DE ATENÇÃO")
    print("="*50)
    
    # Simular diferentes tipos de padrões linguísticos
    tokens = ['O', 'gato', 'subiu', 'no', 'telhado', 'ontem', 'à', 'noite']
    
    # Criar padrões sintáticos simulados
    syntactic_patterns = {
        'determinante-substantivo': np.array([
            [0.1, 0.8, 0.05, 0.02, 0.02, 0.005, 0.005, 0.01],  # "O" -> "gato"
            [0.2, 0.3, 0.1, 0.1, 0.2, 0.05, 0.03, 0.02],      # "gato" distribui atenção
            [0.05, 0.3, 0.2, 0.1, 0.25, 0.05, 0.025, 0.025],  # "subiu" -> sujeito e objeto
            [0.02, 0.05, 0.05, 0.1, 0.75, 0.01, 0.01, 0.01],  # "no" -> "telhado"
            [0.02, 0.2, 0.15, 0.2, 0.3, 0.08, 0.03, 0.02],    # "telhado" 
            [0.01, 0.02, 0.05, 0.02, 0.05, 0.2, 0.1, 0.55],   # "ontem" -> "noite"
            [0.005, 0.01, 0.02, 0.01, 0.05, 0.05, 0.1, 0.755], # "à" -> "noite"
            [0.02, 0.05, 0.08, 0.02, 0.1, 0.4, 0.1, 0.23]     # "noite"
        ])
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Padrão sintático
    im1 = axes[0,0].imshow(syntactic_patterns['determinante-substantivo'], 
                           cmap='YlOrRd', aspect='auto')
    axes[0,0].set_title('Padrão Sintático\n(Determinante-Substantivo)')
    axes[0,0].set_xticks(range(8))
    axes[0,0].set_yticks(range(8))
    axes[0,0].set_xticklabels(tokens, rotation=45)
    axes[0,0].set_yticklabels(tokens)
    plt.colorbar(im1, ax=axes[0,0])
    
    # 2. Grafo de atenção para relações principais
    import networkx as nx
    
    # Criar grafo das principais conexões de atenção
    G = nx.DiGraph()
    attention_matrix = syntactic_patterns['determinante-substantivo']
    
    # Adicionar nós
    for i, token in enumerate(tokens):
        G.add_node(i, label=token)
    
    # Adicionar arestas com pesos significativos (> 0.2)
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if attention_matrix[i, j] > 0.2 and i != j:
                G.add_edge(i, j, weight=attention_matrix[i, j])
    
    # Visualizar grafo
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    axes[0,1].set_title('Grafo de Relações de Atenção\n(conexões > 0.2)')
    
    # Desenhar nós
    nx.draw_networkx_nodes(G, pos, ax=axes[0,1], node_color='lightblue', 
                          node_size=1000, alpha=0.7)
    
    # Desenhar arestas com espessura proporcional ao peso
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, ax=axes[0,1], width=[w*5 for w in weights], 
                          alpha=0.6, edge_color='red', arrows=True, arrowsize=20)
    
    # Desenhar labels
    labels = {i: tokens[i] for i in range(len(tokens))}
    nx.draw_networkx_labels(G, pos, labels, ax=axes[0,1], font_size=10)
    
    axes[0,1].axis('off')
    
    # 3. Análise de distância vs atenção
    distances = []
    attentions = []
    
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if i != j:
                distances.append(abs(i - j))
                attentions.append(attention_matrix[i, j])
    
    axes[0,2].scatter(distances, attentions, alpha=0.6, s=50)
    axes[0,2].set_xlabel('Distância entre Tokens')
    axes[0,2].set_ylabel('Peso de Atenção')
    axes[0,2].set_title('Relação Distância vs Atenção')
    
    # Adicionar linha de tendência
    z = np.polyfit(distances, attentions, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(distances), max(distances), 100)
    axes[0,2].plot(x_trend, p(x_trend), "r--", alpha=0.8, 
                   label=f'Tendência: {z[0]:.3f}x + {z[1]:.3f}')
    axes[0,2].legend()
    
    # 4. Heatmap de tipos de palavras
    word_types = ['Det', 'Subst', 'Verbo', 'Prep', 'Subst', 'Adv', 'Prep', 'Subst']
    
    # Matriz de atenção por tipo de palavra
    type_attention = np.zeros((4, 4))  # Det, Subst, Verbo, Prep, Adv
    type_names = ['Determinante', 'Substantivo', 'Verbo', 'Preposição/Advérbio']
    type_mapping = {'Det': 0, 'Subst': 1, 'Verbo': 2, 'Prep': 3, 'Adv': 3}
    
    type_counts = np.zeros((4, 4))
    
    for i, from_type in enumerate(word_types):
        for j, to_type in enumerate(word_types):
            from_idx = type_mapping[from_type]
            to_idx = type_mapping[to_type]
            type_attention[from_idx, to_idx] += attention_matrix[i, j]
            type_counts[from_idx, to_idx] += 1
    
    # Normalizar pela contagem
    type_attention = np.divide(type_attention, type_counts, 
                              out=np.zeros_like(type_attention), 
                              where=type_counts!=0)
    
    im2 = axes[1,0].imshow(type_attention, cmap='Reds', aspect='auto')
    axes[1,0].set_title('Atenção Média por Tipo de Palavra')
    axes[1,0].set_xticks(range(4))
    axes[1,0].set_yticks(range(4))
    axes[1,0].set_xticklabels(type_names, rotation=45)
    axes[1,0].set_yticklabels(type_names)
    
    # Adicionar valores
    for i in range(4):
        for j in range(4):
            text = axes[1,0].text(j, i, f'{type_attention[i, j]:.2f}',
                                 ha="center", va="center", color="white" if type_attention[i, j] > 0.15 else "black")
    
    plt.colorbar(im2, ax=axes[1,0])
    
    # 5. Análise temporal da atenção
    temporal_tokens = ['ontem', 'à', 'noite']  # Expressão temporal
    temporal_indices = [5, 6, 7]
    
    temporal_attention = attention_matrix[np.ix_(temporal_indices, temporal_indices)]
    
    im3 = axes[1,1].imshow(temporal_attention, cmap='Blues', aspect='auto')
    axes[1,1].set_title('Atenção na Expressão Temporal\n"ontem à noite"')
    axes[1,1].set_xticks(range(3))
    axes[1,1].set_yticks(range(3))
    axes[1,1].set_xticklabels(temporal_tokens)
    axes[1,1].set_yticklabels(temporal_tokens)
    
    for i in range(3):
        for j in range(3):
            text = axes[1,1].text(j, i, f'{temporal_attention[i, j]:.2f}',
                                 ha="center", va="center", color="white" if temporal_attention[i, j] > 0.3 else "black")
    
    plt.colorbar(im3, ax=axes[1,1])
    
    # 6. Distribuição de atenção por posição
    position_attention = np.mean(attention_matrix, axis=0)
    
    axes[1,2].bar(range(8), position_attention, color='green', alpha=0.7)
    axes[1,2].set_title('Atenção Média Recebida\npor Posição')
    axes[1,2].set_xlabel('Posição do Token')
    axes[1,2].set_ylabel('Atenção Média Recebida')
    axes[1,2].set_xticks(range(8))
    axes[1,2].set_xticklabels(tokens, rotation=45)
    
    # Adicionar valores nas barras
    for i, v in enumerate(position_attention):
        axes[1,2].text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return syntactic_patterns

# =============================================================================
# PARTE 8: COMPARAÇÃO COM DIFERENTES ARQUITETURAS
# =============================================================================

def compare_attention_mechanisms():
    """Compara diferentes mecanismos de atenção"""
    print("\n" + "="*50)
    print("COMPARAÇÃO DE MECANISMOS DE ATENÇÃO")
    print("="*50)
    
    tokens = ['O', 'gato', 'subiu', 'no', 'telhado', 'ontem', 'à', 'noite']
    seq_len = len(tokens)
    
    # 1. Self-Attention (como já implementado)
    def self_attention_pattern():
        # Padrão mais realista baseado em dependências sintáticas
        pattern = np.array([
            [0.15, 0.7, 0.1, 0.02, 0.02, 0.005, 0.005, 0.01],
            [0.1, 0.25, 0.4, 0.1, 0.1, 0.02, 0.02, 0.01],
            [0.05, 0.35, 0.2, 0.1, 0.25, 0.02, 0.02, 0.01],
            [0.02, 0.1, 0.1, 0.1, 0.75, 0.01, 0.01, 0.01],
            [0.02, 0.15, 0.2, 0.15, 0.35, 0.08, 0.03, 0.02],
            [0.01, 0.02, 0.05, 0.02, 0.1, 0.3, 0.2, 0.3],
            [0.005, 0.01, 0.02, 0.01, 0.05, 0.1, 0.2, 0.615],
            [0.02, 0.05, 0.08, 0.02, 0.1, 0.25, 0.15, 0.35]
        ])
        return pattern
    
    # 2. Local Attention (janela fixa)
    def local_attention_pattern(window_size=3):
        pattern = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            window_positions = list(range(start, end))
            
            # Distribuir atenção uniformemente na janela
            attention_per_pos = 1.0 / len(window_positions)
            for pos in window_positions:
                pattern[i, pos] = attention_per_pos
        
        return pattern
    
    # 3. Global Attention (todas as posições igualmente)
    def global_attention_pattern():
        return np.ones((seq_len, seq_len)) / seq_len
    
    # 4. Sparse Attention (padrão esparso)
    def sparse_attention_pattern():
        pattern = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            # Atenção para si mesmo
            pattern[i, i] = 0.4
            
            # Atenção para posições específicas (cada 2 posições)
            for j in range(0, seq_len, 2):
                if j != i:
                    pattern[i, j] = 0.6 / (seq_len // 2)
        
        # Normalizar
        pattern = pattern / pattern.sum(axis=1, keepdims=True)
        return pattern
    
    # Gerar padrões
    patterns = {
        'Self-Attention': self_attention_pattern(),
        'Local Attention (w=3)': local_attention_pattern(3),
        'Global Attention': global_attention_pattern(),
        'Sparse Attention': sparse_attention_pattern()
    }
    
    # Visualizar comparação
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (name, pattern) in enumerate(patterns.items()):
        im = axes[i].imshow(pattern, cmap='YlOrRd', aspect='auto')
        axes[i].set_title(f'{name}')
        axes[i].set_xticks(range(8))
        axes[i].set_yticks(range(8))
        axes[i].set_xticklabels(tokens, rotation=45)
        axes[i].set_yticklabels(tokens)
        plt.colorbar(im, ax=axes[i])
    
    plt.suptitle('Comparação de Diferentes Mecanismos de Atenção', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Análise quantitativa
    print("\n📊 ANÁLISE QUANTITATIVA DOS PADRÕES:")
    print("-" * 50)
    
    for name, pattern in patterns.items():
        # Calcular métricas
        sparsity = np.sum(pattern < 0.01) / (seq_len * seq_len)
        max_attention = np.max(pattern)
        entropy = -np.sum(pattern * np.log(pattern + 1e-10), axis=1).mean()
        local_focus = np.mean([pattern[i, max(0, i-1):min(seq_len, i+2)].sum() 
                              for i in range(seq_len)])
        
        print(f"\n{name}:")
        print(f"  - Sparsidade: {sparsity:.2%}")
        print(f"  - Atenção máxima: {max_attention:.3f}")
        print(f"  - Entropia média: {entropy:.3f}")
        print(f"  - Foco local: {local_focus:.3f}")
    
    return patterns

# =============================================================================
# PARTE 9: EXERCÍCIOS INTERATIVOS E DEMONSTRAÇÕES
# =============================================================================

def interactive_attention_demo():
    """Demonstração interativa dos conceitos"""
    print("\n" + "="*50)
    print("DEMONSTRAÇÃO INTERATIVA - CONSTRUINDO INTUIÇÃO")
    print("="*50)
    
    # Exemplo 1: Como a atenção resolve ambiguidade
    print("\n🎯 EXEMPLO 1: RESOLUÇÃO DE AMBIGUIDADE")
    print("Frases: 'O banco do rio' vs 'O banco da cidade'")
    
    sentences = {
        'rio': ['O', 'banco', 'do', 'rio', 'estava', 'cheio', 'de', 'peixes'],
        'cidade': ['O', 'banco', 'da', 'cidade', 'estava', 'cheio', 'de', 'clientes']
    }
    
    # Simular padrões de atenção diferentes para cada contexto
    attention_patterns = {
        'rio': np.array([
            [0.2, 0.6, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005],  # O -> banco
            [0.1, 0.15, 0.2, 0.4, 0.05, 0.05, 0.025, 0.025],  # banco -> rio (contexto)
            [0.05, 0.3, 0.2, 0.4, 0.02, 0.02, 0.005, 0.005],  # do -> rio
            [0.02, 0.25, 0.15, 0.3, 0.1, 0.1, 0.04, 0.04],    # rio -> banco
            [0.05, 0.2, 0.1, 0.15, 0.2, 0.2, 0.05, 0.05],     # estava
            [0.01, 0.1, 0.05, 0.2, 0.15, 0.2, 0.14, 0.15],    # cheio
            [0.005, 0.05, 0.025, 0.1, 0.1, 0.15, 0.2, 0.35],  # de -> peixes
            [0.005, 0.1, 0.025, 0.15, 0.1, 0.2, 0.2, 0.2]     # peixes
        ]),
        'cidade': np.array([
            [0.2, 0.6, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005],  # O -> banco
            [0.1, 0.15, 0.2, 0.4, 0.05, 0.05, 0.025, 0.025],  # banco -> cidade
            [0.05, 0.3, 0.2, 0.4, 0.02, 0.02, 0.005, 0.005],  # da -> cidade
            [0.02, 0.25, 0.15, 0.3, 0.1, 0.1, 0.04, 0.04],    # cidade -> banco
            [0.05, 0.2, 0.1, 0.15, 0.2, 0.2, 0.05, 0.05],     # estava
            [0.01, 0.1, 0.05, 0.2, 0.15, 0.2, 0.14, 0.15],    # cheio
            [0.005, 0.05, 0.025, 0.1, 0.1, 0.15, 0.2, 0.35],  # de -> clientes
            [0.005, 0.1, 0.025, 0.15, 0.1, 0.2, 0.2, 0.2]     # clientes
        ])
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, (context, pattern) in enumerate(attention_patterns.items()):
        im = axes[i].imshow(pattern, cmap='YlOrRd', aspect='auto')
        axes[i].set_title(f'Atenção: "O banco d{{"o" if context=="rio" else "a"}} {context}"')
        axes[i].set_xticks(range(8))
        axes[i].set_yticks(range(8))
        axes[i].set_xticklabels(sentences[context], rotation=45)
        axes[i].set_yticklabels(sentences[context])
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()
    
    # Análise da diferença
    print("📈 ANÁLISE DA DIFERENÇA:")
    print(f"Atenção 'banco'→'rio': {attention_patterns['rio'][1,3]:.3f}")
    print(f"Atenção 'banco'→'cidade': {attention_patterns['cidade'][1,3]:.3f}")
    print(f"Diferença contextual: {abs(attention_patterns['rio'][1,3] - attention_patterns['cidade'][1,3]):.3f}")

def educational_summary():
    """Resumo educacional dos conceitos aprendidos"""
    print("\n" + "="*60)
    print("📚 RESUMO EDUCACIONAL - CONCEITOS PRINCIPAIS")
    print("="*60)
    
    concepts = {
        "1. Embeddings + Positional Encoding": [
            "• Tokens são convertidos em vetores densos (embeddings)",
            "• Posição é codificada usando funções seno/cosseno",
            "• Permite ao modelo entender ordem sem recorrência"
        ],
        
        "2. Self-Attention Mechanism": [
            "• Cada token 'presta atenção' a todos os outros",
            "• Query, Key, Value são projeções lineares do input",
            "• Attention Score = Query · Key^T / √d_k",
            "• Softmax normaliza scores em probabilidades",
            "• Output = Attention_Weights · Values"
        ],
        
        "3. Multi-Head Attention": [
            "• Múltiplas 'cabeças' de atenção em paralelo",
            "• Cada cabeça foca em aspectos diferentes",
            "• Permite capturar relações sintáticas e semânticas",
            "• Outputs são concatenados e projetados"
        ],
        
        "4. Transformer Block": [
            "• Multi-Head Attention + Residual + LayerNorm",
            "• Feed-Forward Network + Residual + LayerNorm",
            "• Permite treinamento profundo e estável",
            "• Base para GPT, BERT, T5, etc."
        ],
        
        "5. Padrões de Atenção": [
            "• Local: Foca em tokens próximos",
            "• Global: Considera toda a sequência",
            "• Sintático: Segue estrutura gramatical",
            "• Semântico: Conecta conceitos relacionados"
        ]
    }
    
    for concept, details in concepts.items():
        print(f"\n{concept}")
        print("-" * len(concept))
        for detail in details:
            print(detail)
    
    print(f"\n{'='*60}")
    print("🎓 PRINCIPAIS INSIGHTS:")
    print("• Attention permite processamento paralelo (vs RNNs)")
    print("• Multi-head captura múltiplos tipos de relações")
    print("• Residual connections permitem redes muito profundas")
    print("• Layer normalization estabiliza o treinamento")
    print("• Transformers são a base dos LLMs modernos")
    print("="*60)

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Executar simulação completa
    run_complete_simulation()
    
    # Análise de interpretabilidade  
    create_attention_interpretation()
    
    # Comparação de arquiteturas
    compare_attention_mechanisms()
    
    # Demonstração interativa
    interactive_attention_demo()
    
    # Resumo educacional
    educational_summary()
    
    print("\n🎉 SIMULAÇÃO COMPLETA FINALIZADA!")
    print("Todos os conceitos dos vídeos DL5 e DL6 foram demonstrados.")
    print("Execute este código no Google Colab para visualizações interativas!")


"""

FUNCIONALIDADES INCLUÍDAS:
✅ Embeddings e Positional Encoding
✅ Self-Attention passo a passo
✅ Multi-Head Attention
✅ Transformer Block completo
✅ Análise de padrões de atenção
✅ Visualizações interpretáveis
✅ Comparação de arquiteturas
✅ Demonstrações interativas
✅ Resumo educacional

TEMPO DE EXECUÇÃO: ~2-3 minutos para gerar todas as visualizações
"""
