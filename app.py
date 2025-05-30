import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
import pandas as pd
import matplotlib.patches as patches
import warnings
import openai
import re
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Simulação de Attention em Transformers",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo personalizado
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4e8df5;
        margin-bottom: 20px;
    }
    .explanation {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .api-key-input {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        margin-bottom: 15px;
    }
    .comparison-container {
        display: flex;
        flex-direction: row;
        gap: 20px;
        margin-bottom: 20px;
    }
    .comparison-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #4e8df5;
        flex: 1;
    }
    .attention-flow {
        background-color: #fff8e1;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("🤖 Simulação de Attention em Transformers")
st.markdown("---")

# Função para gerar frases com a API OpenAI
def generate_sentences_with_openai(api_key, prompt):
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente que gera frases para análise linguística."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Erro ao chamar a API OpenAI: {str(e)}")
        return None

# Função para tokenizar uma frase
def tokenize_sentence(sentence):
    # Tokenização simples por espaço e pontuação
    tokens = re.findall(r'\b\w+\b|[.,!?;]', sentence.lower())
    return tokens[:10]  # Limitar a 10 tokens conforme solicitado

# Sidebar para controles
with st.sidebar:
    st.header("Parâmetros do Modelo")
    
    # Fixar dimensão do modelo em 120 conforme solicitado
    d_model = 120
    st.info(f"Dimensão do Modelo (d_model): {d_model}")
    
    num_heads = st.slider(
        "Número de Cabeças de Atenção",
        min_value=1,
        max_value=16,
        value=8,
        step=1,
        help="Número de cabeças no mecanismo de Multi-Head Attention"
    )
    
    # Fixar comprimento da sequência em 10 conforme solicitado
    seq_length = 10
    st.info(f"Comprimento da Sequência: {seq_length}")
    
    st.markdown("---")
    
    # Entrada da chave da API OpenAI
    st.header("Integração com OpenAI")
    api_key = st.text_input("Chave da API OpenAI", type="password", help="Insira sua chave da API OpenAI para gerar frases comparativas")
    
    if api_key:
        st.success("Chave da API inserida com sucesso!")
    
    st.markdown("---")
    st.markdown("### Sobre esta Aplicação")
    st.markdown("""
    Esta aplicação demonstra o funcionamento do mecanismo de Attention em arquiteturas Transformer.
    
    Agora com integração à API OpenAI para gerar frases comparativas e analisar como o mecanismo de atenção processa diferentes contextos.
    
    Desenvolvido com Streamlit e Matplotlib.
    """)

# Classe principal do simulador
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
    
    def create_embeddings_from_tokens(self, tokens):
        """Cria embeddings para tokens específicos"""
        # Garantir que temos no máximo seq_length tokens
        tokens = tokens[:min(len(tokens), self.seq_length)]
        
        # Preencher com tokens genéricos se necessário
        while len(tokens) < self.seq_length:
            tokens.append(f"<pad>")
            
        embeddings = np.random.randn(self.seq_length, self.d_model) * 0.5
        
        # Adicionar alguma estrutura semântica simulada
        # Isso é uma simplificação, em modelos reais os embeddings seriam aprendidos
        for i, token in enumerate(tokens):
            if token in ["o", "a", "os", "as", "um", "uma"]:  # artigos
                embeddings[i] *= 0.8
            elif token in ["de", "em", "para", "com", "por"]:  # preposições
                embeddings[i] *= 0.7
            elif token in [".", ",", "!", "?"]:  # pontuação
                embeddings[i] *= 0.5
            else:  # substantivos, verbos, etc.
                embeddings[i] += 0.3
        
        return embeddings, tokens
    
    def visualize_embeddings_and_positional(self, tokens):
        """Visualiza embeddings e encoding posicional"""
        embeddings, tokens_used = self.create_embeddings_from_tokens(tokens)
        pos_encoding = self.create_positional_encoding()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Embeddings de tokens
        im1 = axes[0,0].imshow(embeddings.T, cmap='RdBu', aspect='auto')
        axes[0,0].set_title('1. Token Embeddings')
        axes[0,0].set_xlabel('Posição na Sequência')
        axes[0,0].set_ylabel('Dimensões do Embedding')
        axes[0,0].set_xticks(range(self.seq_length))
        axes[0,0].set_xticklabels(tokens_used, rotation=45)
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
        axes[1,0].set_xticks(range(self.seq_length))
        axes[1,0].set_xticklabels(tokens_used, rotation=45)
        plt.colorbar(im3, ax=axes[1,0])
        
        # 4. Padrões do Positional Encoding
        axes[1,1].plot(pos_encoding[:, :10])
        axes[1,1].set_title('4. Padrões Sinusoidais do Positional Encoding')
        axes[1,1].set_xlabel('Posição')
        axes[1,1].set_ylabel('Valor')
        axes[1,1].legend([f'Dim {i}' for i in range(10)], bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        return fig, final_embeddings, tokens_used
    
    def compute_attention_step_by_step(self, X, tokens):
        """Computa self-attention passo a passo com visualizações"""
        # Passo 1: Criar Q, K, V
        Q = X @ self.W_q  # Queries
        K = X @ self.W_k  # Keys  
        V = X @ self.W_v  # Values
        
        # Visualizar Q, K, V
        fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        im1 = axes[0].imshow(Q.T, cmap='Reds', aspect='auto')
        axes[0].set_title('Query Matrix (Q)')
        axes[0].set_xlabel('Tokens')
        axes[0].set_ylabel('Dimensões')
        axes[0].set_xticks(range(self.seq_length))
        axes[0].set_xticklabels(tokens, rotation=45)
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(K.T, cmap='Greens', aspect='auto')
        axes[1].set_title('Key Matrix (K)')
        axes[1].set_xlabel('Tokens')
        axes[1].set_ylabel('Dimensões')
        axes[1].set_xticks(range(self.seq_length))
        axes[1].set_xticklabels(tokens, rotation=45)
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(V.T, cmap='Blues', aspect='auto')
        axes[2].set_title('Value Matrix (V)')
        axes[2].set_xlabel('Tokens')
        axes[2].set_ylabel('Dimensões')
        axes[2].set_xticks(range(self.seq_length))
        axes[2].set_xticklabels(tokens, rotation=45)
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        # Passo 2: Calcular Attention Scores
        attention_scores = Q @ K.T
        
        # Escalar por sqrt(d_k) 
        scaled_scores = attention_scores / np.sqrt(self.d_model)
        
        # Passo 3: Aplicar Softmax
        attention_weights = softmax(scaled_scores, axis=-1)
        
        # Visualizar scores e weights
        fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Attention Scores (antes do softmax)
        im1 = axes[0].imshow(scaled_scores, cmap='RdYlBu', aspect='auto')
        axes[0].set_title('Attention Scores (Escalados)')
        axes[0].set_xlabel('Key Positions')
        axes[0].set_ylabel('Query Positions')
        axes[0].set_xticks(range(self.seq_length))
        axes[0].set_yticks(range(self.seq_length))
        axes[0].set_xticklabels(tokens, rotation=45)
        axes[0].set_yticklabels(tokens)
        plt.colorbar(im1, ax=axes[0])
        
        # Attention Weights (depois do softmax)
        im2 = axes[1].imshow(attention_weights, cmap='YlOrRd', aspect='auto')
        axes[1].set_title('Attention Weights (após Softmax)')
        axes[1].set_xlabel('Key Positions')
        axes[1].set_ylabel('Query Positions')
        axes[1].set_xticks(range(self.seq_length))
        axes[1].set_yticks(range(self.seq_length))
        axes[1].set_xticklabels(tokens, rotation=45)
        axes[1].set_yticklabels(tokens)
        
        # Adicionar valores dos pesos na visualização
        for i in range(self.seq_length):
            for j in range(self.seq_length):
                text = axes[1].text(j, i, f'{attention_weights[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im2, ax=axes[1])
        plt.tight_layout()
        
        # Passo 4: Aplicar pesos aos Values
        output = attention_weights @ V
        
        # Visualizar o output
        fig3, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        im1 = axes[0].imshow(V.T, cmap='Blues', aspect='auto')
        axes[0].set_title('Values Matrix (V)')
        axes[0].set_xlabel('Tokens')
        axes[0].set_ylabel('Dimensões')
        axes[0].set_xticks(range(self.seq_length))
        axes[0].set_xticklabels(tokens, rotation=45)
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(output.T, cmap='Purples', aspect='auto')
        axes[1].set_title('Attention Output')
        axes[1].set_xlabel('Tokens')
        axes[1].set_ylabel('Dimensões')
        axes[1].set_xticks(range(self.seq_length))
        axes[1].set_xticklabels(tokens, rotation=45)
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        return fig1, fig2, fig3, Q, K, V, attention_weights, output
    
    def visualize_attention_flow(self, attention_weights, tokens, token_index=3):
        """Visualiza o fluxo de atenção para um token específico"""
        # Garantir que o índice está dentro dos limites
        token_index = min(token_index, len(tokens)-1)
        
        # Obter os pesos de atenção para o token selecionado
        token_attention = attention_weights[token_index, :]
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Posições dos tokens no eixo x
        token_positions = np.arange(len(tokens))
        
        # Altura das barras (pesos de atenção)
        bar_heights = token_attention
        
        # Criar barras coloridas
        bars = ax.bar(token_positions, bar_heights, color='skyblue', alpha=0.7)
        
        # Destacar o token atual
        bars[token_index].set_color('red')
        bars[token_index].set_alpha(1.0)
        
        # Adicionar setas para tokens anteriores com peso significativo
        for i in range(token_index):
            if token_attention[i] > 0.05:  # Apenas setas para tokens com peso significativo
                # Coordenadas para a seta
                start_x = token_positions[token_index]
                start_y = bar_heights[token_index] * 0.8
                end_x = token_positions[i]
                end_y = bar_heights[i] * 0.8
                
                # Desenhar seta
                ax.annotate('', 
                            xy=(end_x, end_y), 
                            xytext=(start_x, start_y),
                            arrowprops=dict(arrowstyle='->', 
                                           lw=2, 
                                           color='darkred', 
                                           alpha=min(1.0, token_attention[i]*3)),
                            )
        
        # Adicionar valores nas barras
        for i, v in enumerate(bar_heights):
            ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Configurar eixos
        ax.set_xticks(token_positions)
        ax.set_xticklabels(tokens, rotation=45)
        ax.set_ylabel('Peso de Atenção')
        ax.set_title(f'Fluxo de Atenção para o Token "{tokens[token_index]}"')
        
        # Adicionar texto explicativo
        ax.text(0.5, 0.95, 
                f'O token "{tokens[token_index]}" analisa todos os tokens anteriores\npara construir sua representação contextualizada',
                transform=ax.transAxes, 
                ha='center', 
                va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        
        return fig
    
    def analyze_attention_patterns(self, attention_weights, tokens):
        """Analisa padrões específicos de atenção"""
        # 1. Atenção por token específico
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Análise para o primeiro substantivo (posição 1)
        token_pos1 = min(1, self.seq_length-1)
        token1_attention = attention_weights[token_pos1, :]
        axes[0,0].bar(range(self.seq_length), token1_attention, color='skyblue', alpha=0.7)
        axes[0,0].set_title(f'Atenção de "{tokens[token_pos1]}" para outros tokens')
        axes[0,0].set_xlabel('Tokens')
        axes[0,0].set_ylabel('Peso de Atenção')
        axes[0,0].set_xticks(range(self.seq_length))
        axes[0,0].set_xticklabels(tokens, rotation=45)
        
        # Adicionar valores nas barras
        for i, v in enumerate(token1_attention):
            axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Análise para o verbo (posição 2)
        token_pos2 = min(2, self.seq_length-1)
        token2_attention = attention_weights[token_pos2, :]
        axes[0,1].bar(range(self.seq_length), token2_attention, color='lightcoral', alpha=0.7)
        axes[0,1].set_title(f'Atenção de "{tokens[token_pos2]}" para outros tokens')
        axes[0,1].set_xlabel('Tokens')
        axes[0,1].set_ylabel('Peso de Atenção')
        axes[0,1].set_xticks(range(self.seq_length))
        axes[0,1].set_xticklabels(tokens, rotation=45)
        
        for i, v in enumerate(token2_attention):
            axes[0,1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Mapa de calor com anotações
        sns.heatmap(attention_weights, annot=True, fmt='.2f', 
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
        
        return fig
    
    def compare_attention_patterns(self, attention_weights1, tokens1, attention_weights2, tokens2):
        """Compara padrões de atenção entre duas frases"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # Mapa de calor para a primeira frase
        sns.heatmap(attention_weights1, annot=True, fmt='.2f', 
                   xticklabels=tokens1, yticklabels=tokens1,
                   cmap='YlOrRd', ax=axes[0,0])
        axes[0,0].set_title('Matriz de Atenção - Frase 1')
        axes[0,0].set_xlabel('Attending to (Keys)')
        axes[0,0].set_ylabel('Attending from (Queries)')
        
        # Mapa de calor para a segunda frase
        sns.heatmap(attention_weights2, annot=True, fmt='.2f', 
                   xticklabels=tokens2, yticklabels=tokens2,
                   cmap='YlOrRd', ax=axes[0,1])
        axes[0,1].set_title('Matriz de Atenção - Frase 2')
        axes[0,1].set_xlabel('Attending to (Keys)')
        axes[0,1].set_ylabel('Attending from (Queries)')
        
        # Distribuição dos pesos para a primeira frase
        axes[1,0].hist(attention_weights1.flatten(), bins=20, alpha=0.7, color='blue', label='Frase 1')
        axes[1,0].set_title('Distribuição dos Pesos - Frase 1')
        axes[1,0].set_xlabel('Valor do Peso')
        axes[1,0].set_ylabel('Frequência')
        axes[1,0].axvline(attention_weights1.mean(), color='darkblue', linestyle='--', 
                         label=f'Média: {attention_weights1.mean():.3f}')
        axes[1,0].legend()
        
        # Distribuição dos pesos para a segunda frase
        axes[1,1].hist(attention_weights2.flatten(), bins=20, alpha=0.7, color='red', label='Frase 2')
        axes[1,1].set_title('Distribuição dos Pesos - Frase 2')
        axes[1,1].set_xlabel('Valor do Peso')
        axes[1,1].set_ylabel('Frequência')
        axes[1,1].axvline(attention_weights2.mean(), color='darkred', linestyle='--', 
                         label=f'Média: {attention_weights2.mean():.3f}')
        axes[1,1].legend()
        
        plt.tight_layout()
        
        return fig

class MultiHeadAttention:
    def __init__(self, d_model=64, num_heads=8, seq_length=8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.seq_length = seq_length
        
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
    
    def compute_multi_head_attention(self, X, tokens):
        """Computa multi-head attention"""
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
        
        # Visualizar diferentes cabeças
        fig = self.visualize_multi_head_patterns(head_attentions, tokens)
        
        return fig, final_output, head_attentions
    
    def visualize_multi_head_patterns(self, head_attentions, tokens):
        """Visualiza padrões de atenção de diferentes cabeças"""
        # Determinar o layout da figura com base no número de cabeças
        if self.num_heads <= 4:
            nrows, ncols = 1, self.num_heads
        elif self.num_heads <= 8:
            nrows, ncols = 2, 4
        else:
            nrows, ncols = (self.num_heads + 3) // 4, 4
            
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5*nrows))
        
        if self.num_heads == 1:
            axes = [axes]  # Converter para lista para indexação consistente
        else:
            axes = axes.flatten()
        
        for i in range(self.num_heads):
            if i < len(axes):
                im = axes[i].imshow(head_attentions[i], cmap='YlOrRd', aspect='auto')
                axes[i].set_title(f'Cabeça {i+1}')
                axes[i].set_xticks(range(self.seq_length))
                axes[i].set_yticks(range(self.seq_length))
                axes[i].set_xticklabels(tokens, rotation=45, fontsize=8)
                axes[i].set_yticklabels(tokens, fontsize=8)
                plt.colorbar(im, ax=axes[i])
        
        # Ocultar eixos não utilizados
        for i in range(self.num_heads, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Padrões de Atenção por Cabeça', fontsize=16)
        plt.tight_layout()
        
        return fig

# Função principal
def main():
    # Introdução
    st.markdown("""
    <div class="highlight">
        <h2>Entendendo o Mecanismo de Attention em Transformers</h2>
        <p>Esta aplicação demonstra visualmente como funciona o mecanismo de Attention, 
        componente fundamental das arquiteturas Transformer que revolucionaram o Processamento 
        de Linguagem Natural e outras áreas de IA.</p>
        <p>Agora com suporte para comparação de duas frases geradas pela API OpenAI!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Seção para geração de frases com OpenAI
    st.header("Geração de Frases para Comparação")
    
    st.markdown("""
    <div class="explanation">
        <p>Insira sua chave da API OpenAI no painel lateral para gerar duas frases comparativas. 
        Estas frases serão usadas para demonstrar como o mecanismo de atenção processa diferentes contextos.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Opção para usar frases de exemplo ou gerar com OpenAI
    use_example = st.checkbox("Usar frases de exemplo (sem API OpenAI)", value=not bool(api_key))
    
    if use_example:
        # Frases de exemplo
        sentence1 = "O gato de botas caminha pela floresta"
        sentence2 = "A galinha bota ovos no galinheiro"
        
        st.markdown("""
        <div class="comparison-container">
            <div class="comparison-card">
                <h4>Frase 1:</h4>
                <p>O gato de botas caminha pela floresta</p>
            </div>
            <div class="comparison-card">
                <h4>Frase 2:</h4>
                <p>A galinha bota ovos no galinheiro</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if api_key:
            # Botão para gerar frases
            if st.button("Gerar Frases Comparativas"):
                with st.spinner("Gerando frases com a API OpenAI..."):
                    # Prompt para gerar frases comparativas
                    prompt = """
                    Gere duas frases em português que contenham palavras homônimas (mesma grafia, significados diferentes).
                    Por exemplo, palavras como "gato" (animal) e "bota" (calçado) versus "bota" (verbo botar).
                    Forneça apenas as duas frases, sem explicações adicionais.
                    Cada frase deve ter no máximo 10 palavras.
                    """
                    
                    result = generate_sentences_with_openai(api_key, prompt)
                    
                    if result:
                        # Dividir o resultado em duas frases
                        sentences = result.split('\n')
                        if len(sentences) >= 2:
                            sentence1 = sentences[0].strip()
                            sentence2 = sentences[1].strip()
                            
                            st.session_state['sentence1'] = sentence1
                            st.session_state['sentence2'] = sentence2
                            
                            st.markdown(f"""
                            <div class="comparison-container">
                                <div class="comparison-card">
                                    <h4>Frase 1:</h4>
                                    <p>{sentence1}</p>
                                </div>
                                <div class="comparison-card">
                                    <h4>Frase 2:</h4>
                                    <p>{sentence2}</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("Não foi possível obter duas frases distintas da API.")
            elif 'sentence1' in st.session_state and 'sentence2' in st.session_state:
                # Mostrar frases já geradas
                sentence1 = st.session_state['sentence1']
                sentence2 = st.session_state['sentence2']
                
                st.markdown(f"""
                <div class="comparison-container">
                    <div class="comparison-card">
                        <h4>Frase 1:</h4>
                        <p>{sentence1}</p>
                    </div>
                    <div class="comparison-card">
                        <h4>Frase 2:</h4>
                        <p>{sentence2}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Mensagem para clicar no botão
                st.info("Clique no botão 'Gerar Frases Comparativas' para obter frases da API OpenAI.")
                # Usar frases de exemplo como fallback
                sentence1 = "O gato de botas caminha pela floresta"
                sentence2 = "A galinha bota ovos no galinheiro"
        else:
            # Mensagem para inserir a chave da API
            st.warning("Insira sua chave da API OpenAI no painel lateral ou use as frases de exemplo.")
            # Usar frases de exemplo como fallback
            sentence1 = "O gato de botas caminha pela floresta"
            sentence2 = "A galinha bota ovos no galinheiro"
    
    # Tokenizar as frases
    tokens1 = tokenize_sentence(sentence1)
    tokens2 = tokenize_sentence(sentence2)
    
    # Inicializar o simulador com os parâmetros do usuário
    simulator = TransformerSimulator(d_model=d_model, seq_length=seq_length)
    
    st.markdown("---")
    
    # Parte 1: Embeddings e Positional Encoding
    st.header("1. Embeddings e Positional Encoding")
    
    st.markdown("""
    <div class="explanation">
        <p>Em um modelo Transformer, as palavras (tokens) são primeiro convertidas em vetores densos chamados <b>embeddings</b>. 
        Como os Transformers processam todos os tokens simultaneamente (e não sequencialmente como RNNs), 
        precisamos adicionar informação sobre a posição de cada token na sequência. Isso é feito através do <b>Positional Encoding</b>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Criar abas para as duas frases
    tab1, tab2 = st.tabs(["Frase 1", "Frase 2"])
    
    with tab1:
        fig1, embeddings1, tokens_used1 = simulator.visualize_embeddings_and_positional(tokens1)
        st.pyplot(fig1)
    
    with tab2:
        fig2, embeddings2, tokens_used2 = simulator.visualize_embeddings_and_positional(tokens2)
        st.pyplot(fig2)
    
    st.markdown("""
    <div class="explanation">
        <p><b>Explicação dos gráficos:</b></p>
        <ol>
            <li><b>Token Embeddings</b>: Representação vetorial de cada palavra.</li>
            <li><b>Positional Encoding</b>: Informação sobre a posição de cada token na sequência.</li>
            <li><b>Embeddings Finais</b>: Combinação dos embeddings de token com o encoding posicional.</li>
            <li><b>Padrões Sinusoidais</b>: Visualização das funções seno/cosseno usadas no encoding posicional.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Parte 2: Self-Attention Passo a Passo
    st.header("2. Mecanismo de Self-Attention Passo a Passo")
    
    st.markdown("""
    <div class="explanation">
        <p>O mecanismo de <b>Self-Attention</b> permite que cada token "preste atenção" a todos os outros tokens da sequência, 
        capturando relações de longo alcance. Vamos ver como isso funciona passo a passo:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Criar abas para as duas frases
    tab1, tab2 = st.tabs(["Frase 1", "Frase 2"])
    
    with tab1:
        # Passo 1: Criar Q, K, V
        st.subheader("🔍 Passo 1: Criando Query, Key e Value matrices")
        
        st.markdown("""
        <div class="explanation">
            <p>Para cada token, criamos três vetores diferentes:</p>
            <ul>
                <li><b>Query (Q)</b>: O que o token está "perguntando" ou "procurando"</li>
                <li><b>Key (K)</b>: O que o token está "oferecendo" ou "respondendo"</li>
                <li><b>Value (V)</b>: A informação real que será agregada</li>
            </ul>
            <p>Estes vetores são criados através de transformações lineares (multiplicação por matrizes de peso) dos embeddings.</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig_qkv1, fig_scores1, fig_output1, Q1, K1, V1, attention_weights1, output1 = simulator.compute_attention_step_by_step(embeddings1, tokens_used1)
        st.pyplot(fig_qkv1)
        
        # Passo 2 e 3: Calcular Scores e Aplicar Softmax
        st.subheader("🧮 Passo 2 & 3: Calculando Attention Scores e Aplicando Softmax")
        
        st.markdown(f"""
        <div class="explanation">
            <p><b>Passo 2:</b> Calculamos os "scores" de atenção multiplicando Q por K transposto e escalando pelo fator 1/√{d_model} = {1/np.sqrt(d_model):.3f}</p>
            <p><b>Passo 3:</b> Aplicamos a função softmax para converter os scores em pesos de atenção que somam 1 para cada token.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.pyplot(fig_scores1)
        
        # Passo 4: Aplicar pesos aos Values
        st.subheader("🎯 Passo 4: Computando Output (Attention × Values)")
        
        st.markdown("""
        <div class="explanation">
            <p>Finalmente, multiplicamos os pesos de atenção pela matriz V para obter a saída do mecanismo de atenção.
            Cada token agora contém informação ponderada de todos os outros tokens da sequência.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.pyplot(fig_output1)
        
        # Visualização do fluxo de atenção
        st.subheader("🔄 Fluxo de Atenção para Tokens Específicos")
        
        st.markdown("""
        <div class="attention-flow">
            <p>Uma característica fundamental do mecanismo de atenção é que <b>cada token considera o conjunto de tokens anteriores</b> 
            para construir sua representação contextualizada. Isso permite que o modelo capture relações semânticas e sintáticas entre palavras.</p>
            <p>O gráfico abaixo mostra como um token específico "presta atenção" aos tokens anteriores, com setas indicando as conexões mais fortes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Seletor para o token a ser analisado
        token_to_analyze1 = st.slider("Selecione o token para analisar (Frase 1)", 
                                     min_value=2, 
                                     max_value=min(len(tokens_used1)-1, 9), 
                                     value=3,
                                     help="Escolha um token para visualizar como ele presta atenção aos tokens anteriores")
        
        flow_fig1 = simulator.visualize_attention_flow(attention_weights1, tokens_used1, token_to_analyze1)
        st.pyplot(flow_fig1)
    
    with tab2:
        # Passo 1: Criar Q, K, V
        st.subheader("🔍 Passo 1: Criando Query, Key e Value matrices")
        
        st.markdown("""
        <div class="explanation">
            <p>Para cada token, criamos três vetores diferentes:</p>
            <ul>
                <li><b>Query (Q)</b>: O que o token está "perguntando" ou "procurando"</li>
                <li><b>Key (K)</b>: O que o token está "oferecendo" ou "respondendo"</li>
                <li><b>Value (V)</b>: A informação real que será agregada</li>
            </ul>
            <p>Estes vetores são criados através de transformações lineares (multiplicação por matrizes de peso) dos embeddings.</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig_qkv2, fig_scores2, fig_output2, Q2, K2, V2, attention_weights2, output2 = simulator.compute_attention_step_by_step(embeddings2, tokens_used2)
        st.pyplot(fig_qkv2)
        
        # Passo 2 e 3: Calcular Scores e Aplicar Softmax
        st.subheader("🧮 Passo 2 & 3: Calculando Attention Scores e Aplicando Softmax")
        
        st.markdown(f"""
        <div class="explanation">
            <p><b>Passo 2:</b> Calculamos os "scores" de atenção multiplicando Q por K transposto e escalando pelo fator 1/√{d_model} = {1/np.sqrt(d_model):.3f}</p>
            <p><b>Passo 3:</b> Aplicamos a função softmax para converter os scores em pesos de atenção que somam 1 para cada token.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.pyplot(fig_scores2)
        
        # Passo 4: Aplicar pesos aos Values
        st.subheader("🎯 Passo 4: Computando Output (Attention × Values)")
        
        st.markdown("""
        <div class="explanation">
            <p>Finalmente, multiplicamos os pesos de atenção pela matriz V para obter a saída do mecanismo de atenção.
            Cada token agora contém informação ponderada de todos os outros tokens da sequência.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.pyplot(fig_output2)
        
        # Visualização do fluxo de atenção
        st.subheader("🔄 Fluxo de Atenção para Tokens Específicos")
        
        st.markdown("""
        <div class="attention-flow">
            <p>Uma característica fundamental do mecanismo de atenção é que <b>cada token considera o conjunto de tokens anteriores</b> 
            para construir sua representação contextualizada. Isso permite que o modelo capture relações semânticas e sintáticas entre palavras.</p>
            <p>O gráfico abaixo mostra como um token específico "presta atenção" aos tokens anteriores, com setas indicando as conexões mais fortes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Seletor para o token a ser analisado
        token_to_analyze2 = st.slider("Selecione o token para analisar (Frase 2)", 
                                     min_value=2, 
                                     max_value=min(len(tokens_used2)-1, 9), 
                                     value=3,
                                     help="Escolha um token para visualizar como ele presta atenção aos tokens anteriores")
        
        flow_fig2 = simulator.visualize_attention_flow(attention_weights2, tokens_used2, token_to_analyze2)
        st.pyplot(flow_fig2)
    
    st.markdown("---")
    
    # Parte 3: Análise Comparativa de Padrões de Atenção
    st.header("3. Análise Comparativa de Padrões de Atenção")
    
    st.markdown("""
    <div class="explanation">
        <p>Vamos comparar os padrões de atenção que emergem nas duas frases. Esta comparação nos permite ver como o mecanismo de atenção
        se comporta de forma diferente dependendo do contexto e da estrutura da frase.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Análise individual
    tab1, tab2 = st.tabs(["Análise Individual", "Comparação Direta"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Frase 1")
            fig_patterns1 = simulator.analyze_attention_patterns(attention_weights1, tokens_used1)
            st.pyplot(fig_patterns1)
        
        with col2:
            st.subheader("Frase 2")
            fig_patterns2 = simulator.analyze_attention_patterns(attention_weights2, tokens_used2)
            st.pyplot(fig_patterns2)
    
    with tab2:
        st.subheader("Comparação Direta dos Padrões de Atenção")
        fig_comparison = simulator.compare_attention_patterns(attention_weights1, tokens_used1, attention_weights2, tokens_used2)
        st.pyplot(fig_comparison)
        
        st.markdown("""
        <div class="explanation">
            <p><b>Observações sobre as diferenças:</b></p>
            <ul>
                <li>Note como palavras homônimas (como "bota" - verbo vs. calçado) apresentam padrões de atenção diferentes dependendo do contexto</li>
                <li>A distribuição dos pesos de atenção varia entre as frases, refletindo suas diferentes estruturas sintáticas</li>
                <li>Tokens em posições semelhantes podem ter comportamentos de atenção muito diferentes dependendo do seu papel na frase</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Parte 4: Multi-Head Attention
    st.header("4. Multi-Head Attention")
    
    st.markdown("""
    <div class="explanation">
        <p>Em vez de ter apenas um mecanismo de atenção, os Transformers usam <b>múltiplas cabeças de atenção</b> em paralelo.
        Cada cabeça pode se especializar em diferentes tipos de relações entre tokens.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializar Multi-Head Attention
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, seq_length=seq_length)
    
    # Computar e visualizar
    tab1, tab2 = st.tabs(["Frase 1", "Frase 2"])
    
    with tab1:
        fig_mha1, final_output1, head_attentions1 = mha.compute_multi_head_attention(embeddings1, tokens_used1)
        
        st.markdown(f"""
        <div class="explanation">
            <p><b>Configuração atual:</b></p>
            <ul>
                <li>Número de cabeças: {num_heads}</li>
                <li>Dimensão por cabeça (d_k): {d_model // num_heads}</li>
                <li>Dimensão total do modelo (d_model): {d_model}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.pyplot(fig_mha1)
    
    with tab2:
        fig_mha2, final_output2, head_attentions2 = mha.compute_multi_head_attention(embeddings2, tokens_used2)
        
        st.markdown(f"""
        <div class="explanation">
            <p><b>Configuração atual:</b></p>
            <ul>
                <li>Número de cabeças: {num_heads}</li>
                <li>Dimensão por cabeça (d_k): {d_model // num_heads}</li>
                <li>Dimensão total do modelo (d_model): {d_model}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.pyplot(fig_mha2)
    
    st.markdown("""
    <div class="explanation">
        <p>Cada cabeça de atenção captura diferentes aspectos das relações entre tokens. 
        Algumas podem focar em relações gramaticais, outras em relações semânticas ou de proximidade.</p>
        <p>Os outputs de todas as cabeças são concatenados e passados por uma transformação linear final 
        para produzir a saída do bloco de Multi-Head Attention.</p>
        <p>Observe como as diferentes cabeças apresentam padrões distintos para as duas frases, 
        demonstrando como o modelo captura diferentes aspectos do contexto.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Conclusão
    st.markdown("---")
    st.header("Conclusão")
    
    st.markdown("""
    <div class="highlight">
        <p>O mecanismo de Attention é o componente central que permite aos Transformers capturar relações complexas 
        entre elementos de uma sequência, independentemente da distância entre eles.</p>
        
        <p>Como vimos na comparação entre as duas frases, o contexto é fundamental para a interpretação das palavras. 
        Cada token <b>analisa e considera o conjunto de palavras anteriores</b> para construir sua representação, 
        permitindo que o modelo diferencie palavras homônimas e capture relações semânticas sutis.</p>
        
        <p>Esta capacidade revolucionou o Processamento de Linguagem Natural e outras áreas de IA, 
        permitindo o desenvolvimento de modelos como BERT, GPT, T5 e outros que alcançam resultados 
        impressionantes em diversas tarefas.</p>
        
        <p>Experimente ajustar o número de cabeças de atenção no painel lateral e gerar diferentes pares de frases 
        para ver como o mecanismo de atenção se adapta a diferentes contextos!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
