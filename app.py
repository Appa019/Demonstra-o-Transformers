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

st.set_page_config(
    page_title="Simulação de Attention em Transformers",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
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
    .llm-explanation {
        background-color: #e8f5e8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin-bottom: 20px;
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
        flex-wrap: wrap;
    }
    .comparison-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4e8df5;
        flex: 1;
        min-width: 300px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .comparison-card h4 {
        margin-top: 0;
        color: #2c3e50;
        font-size: 1.1em;
    }
    .comparison-card p {
        font-size: 1.1em;
        line-height: 1.4;
        margin-bottom: 0;
        color: #34495e;
        font-weight: 500;
    }
    .attention-flow {
        background-color: #fff8e1;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin-bottom: 15px;
    }
    .importance-analysis {
        background-color: #f3e5f5;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #9c27b0;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Simulação de Attention em Transformers")
st.markdown("---")

def generate_sentences_with_openai(api_key, prompt):
    """Gera frases usando a API OpenAI com tratamento de erros robusto"""
    try:
        if not api_key or not api_key.strip():
            raise ValueError("Chave da API não fornecida")
        
        if not api_key.startswith('sk-'):
            raise ValueError("Formato da chave inválido - deve começar com 'sk-'")
        
        client = openai.OpenAI(api_key=api_key.strip())
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "Você é um assistente especializado em gerar frases para análise linguística de modelos de linguagem. Sempre gere exatamente duas frases conforme solicitado."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
            timeout=30  # Timeout de 30 segundos
        )
        
        return response.choices[0].message.content.strip()
        
    except openai.AuthenticationError:
        raise Exception("❌ Erro de autenticação: Verifique se sua chave da API está correta")
    except openai.RateLimitError:
        raise Exception("⏱️ Limite de taxa excedido: Aguarde alguns minutos e tente novamente")
    except openai.APIConnectionError:
        raise Exception("🌐 Erro de conexão: Verifique sua conexão com a internet")
    except openai.APITimeoutError:
        raise Exception("⏱️ Timeout: A API demorou muito para responder")
    except Exception as e:
        if "api_key" in str(e).lower():
            raise Exception("🔑 Erro na chave da API: Verifique se ela está correta e ativa")
        else:
            raise Exception(f"❌ Erro inesperado: {str(e)}")


def tokenize_sentence(sentence):
    # Tokenização simples por espaço e pontuação
    tokens = re.findall(r'\b\w+\b|[.,!?;]', sentence.lower())
    return tokens[:10]  # Limitar a 10 tokens conforme solicitado

def calculate_token_importance(attention_weights, tokens, real_tokens_count):
    """Calcula a importância de cada token baseada nos pesos de atenção"""
    # Usar apenas tokens reais
    real_attention = attention_weights[:real_tokens_count, :real_tokens_count]
    
    # Importância como soma dos pesos de atenção recebidos (quanto outros tokens prestam atenção a este)
    importance_received = np.sum(real_attention, axis=0)
    
    # Importância como soma dos pesos de atenção dados (quanto este token presta atenção aos outros)
    importance_given = np.sum(real_attention, axis=1)
    
    # Importância combinada (média das duas métricas normalizadas)
    importance_combined = (importance_received + importance_given) / 2
    
    return importance_received, importance_given, importance_combined

with st.sidebar:
    st.header("⚙️ Parâmetros do Modelo")
    
    # Fixar dimensão do modelo em 40 conforme solicitado
    d_model = 40
    st.info(f"📏 Dimensão do Modelo (d_model): {d_model}")
    
    num_heads = st.slider(
        "🧠 Número de Cabeças de Atenção",
        min_value=1,
        max_value=12,
        value=8,
        step=1,
        help="Número de cabeças no mecanismo de Multi-Head Attention"
    )
    
    if d_model % num_heads != 0:
        st.warning(f"⚠️ Para evitar erros, o número de cabeças deve ser um divisor de d_model ({d_model}).")
        valid_heads = [h for h in range(1, 13) if d_model % h == 0]
        st.info(f"✅ Valores válidos para número de cabeças: {', '.join(map(str, valid_heads))}")
        # Escolher o valor válido mais próximo
        if num_heads not in valid_heads:
            num_heads = max([h for h in valid_heads if h <= num_heads], default=valid_heads[0])
            st.success(f"🔧 Ajustado para {num_heads} cabeças.")
    
    seq_length = 10
    st.info(f"📐 Comprimento da Sequência: {seq_length}")
    
    st.markdown("---")
    
    st.header("🔑 Integração com OpenAI")
    
    api_key = st.text_input(
        "Chave da API OpenAI", 
        type="password", 
        help="Insira sua chave da API OpenAI para gerar frases comparativas",
        placeholder="sk-..."
    )
    
    if api_key:
        if api_key.startswith('sk-') and len(api_key) > 20:
            st.success("✅ Chave da API inserida com sucesso!")
        else:
            st.warning("⚠️ Formato da chave parece incorreto. Certifique-se de que começa com 'sk-'")
    else:
        st.info("💡 Cole sua chave da API OpenAI acima para gerar frases personalizadas")
    
    st.markdown("---")
    
    st.header("📚 Sobre esta Aplicação")
    st.markdown("""
    **🎯 Objetivo:**
    Demonstrar visualmente como funciona o mecanismo de Attention em arquiteturas Transformer.
    
    **✨ Funcionalidades:**
    - 🔍 Análise passo a passo do Self-Attention
    - 🧠 Visualização Multi-Head Attention  
    - ⚖️ Análise de importância de tokens
    - 🔄 Comparação entre frases
    - 🤖 Integração com API OpenAI
    
    **🛠️ Tecnologias:**
    Streamlit • Matplotlib • NumPy • OpenAI API
    """)

class TransformerSimulator:
    def __init__(self, d_model=64, seq_length=8, vocab_size=1000):
        self.d_model = d_model
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
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
        
        real_tokens_count = len(tokens)
        
        while len(tokens) < self.seq_length:
            tokens.append("")  # Usar string vazia em vez de <pad> para não poluir visualização
            
        embeddings = np.random.randn(self.seq_length, self.d_model) * 0.5
        
        for i, token in enumerate(tokens):
            if i >= real_tokens_count:  # Para tokens de padding
                embeddings[i] *= 0.1  # Valores menores para padding
            elif token in ["o", "a", "os", "as", "um", "uma"]:  # artigos
                embeddings[i] *= 0.8
            elif token in ["de", "em", "para", "com", "por"]:  # preposições
                embeddings[i] *= 0.7
            elif token in [".", ",", "!", "?"]:  # pontuação
                embeddings[i] *= 0.5
            else:  # substantivos, verbos, etc.
                embeddings[i] += 0.3
        
        return embeddings, tokens, real_tokens_count
    
    def visualize_embeddings_and_positional(self, tokens):
        """Visualiza embeddings e encoding posicional"""
        embeddings, tokens_used, real_tokens_count = self.create_embeddings_from_tokens(tokens)
        pos_encoding = self.create_positional_encoding()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        x_labels = [t if t else " " for t in tokens_used]
        
        im1 = axes[0,0].imshow(embeddings.T, cmap='RdBu', aspect='auto')
        axes[0,0].set_title('1. Token Embeddings')
        axes[0,0].set_xlabel('Posição na Sequência')
        axes[0,0].set_ylabel('Dimensões do Embedding')
        axes[0,0].set_xticks(range(self.seq_length))
        axes[0,0].set_xticklabels(x_labels, rotation=45, fontsize=9)
        
        for i in range(real_tokens_count, self.seq_length):
            axes[0,0].axvline(x=i-0.5, color='gray', linestyle='--', alpha=0.3)
            
        plt.colorbar(im1, ax=axes[0,0])
        
        im2 = axes[0,1].imshow(pos_encoding.T, cmap='viridis', aspect='auto')
        axes[0,1].set_title('2. Positional Encoding')
        axes[0,1].set_xlabel('Posição na Sequência')
        axes[0,1].set_ylabel('Dimensões do Encoding')
        plt.colorbar(im2, ax=axes[0,1])
        
        final_embeddings = embeddings + pos_encoding
        im3 = axes[1,0].imshow(final_embeddings.T, cmap='RdBu', aspect='auto')
        axes[1,0].set_title('3. Embeddings Finais (Token + Posicional)')
        axes[1,0].set_xlabel('Posição na Sequência')
        axes[1,0].set_ylabel('Dimensões')
        axes[1,0].set_xticks(range(self.seq_length))
        axes[1,0].set_xticklabels(x_labels, rotation=45, fontsize=9)
        
        for i in range(real_tokens_count, self.seq_length):
            axes[1,0].axvline(x=i-0.5, color='gray', linestyle='--', alpha=0.3)
            
        plt.colorbar(im3, ax=axes[1,0])
        
        axes[1,1].plot(pos_encoding[:, :10])
        axes[1,1].set_title('4. Padrões Sinusoidais do Positional Encoding')
        axes[1,1].set_xlabel('Posição')
        axes[1,1].set_ylabel('Valor')
        axes[1,1].legend([f'Dim {i}' for i in range(10)], loc='upper right', fontsize=8)
        
        plt.tight_layout()
        
        return fig, final_embeddings, tokens_used, real_tokens_count
    
    def compute_attention_step_by_step(self, X, tokens, real_tokens_count):
        """Computa self-attention passo a passo com visualizações"""
        # Passo 1: Criar Q, K, V
        Q = X @ self.W_q  # Queries
        K = X @ self.W_k  # Keys  
        V = X @ self.W_v  # Values
        
        x_labels = [t if t else " " for t in tokens]
        
        fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        im1 = axes[0].imshow(Q.T, cmap='Reds', aspect='auto')
        axes[0].set_title('Query Matrix (Q)')
        axes[0].set_xlabel('Tokens')
        axes[0].set_ylabel('Dimensões')
        axes[0].set_xticks(range(self.seq_length))
        axes[0].set_xticklabels(x_labels, rotation=45, fontsize=9)
        
        for i in range(real_tokens_count, self.seq_length):
            axes[0].axvline(x=i-0.5, color='gray', linestyle='--', alpha=0.3)
            
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(K.T, cmap='Greens', aspect='auto')
        axes[1].set_title('Key Matrix (K)')
        axes[1].set_xlabel('Tokens')
        axes[1].set_ylabel('Dimensões')
        axes[1].set_xticks(range(self.seq_length))
        axes[1].set_xticklabels(x_labels, rotation=45, fontsize=9)
        
        for i in range(real_tokens_count, self.seq_length):
            axes[1].axvline(x=i-0.5, color='gray', linestyle='--', alpha=0.3)
            
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(V.T, cmap='Blues', aspect='auto')
        axes[2].set_title('Value Matrix (V)')
        axes[2].set_xlabel('Tokens')
        axes[2].set_ylabel('Dimensões')
        axes[2].set_xticks(range(self.seq_length))
        axes[2].set_xticklabels(x_labels, rotation=45, fontsize=9)
        
        for i in range(real_tokens_count, self.seq_length):
            axes[2].axvline(x=i-0.5, color='gray', linestyle='--', alpha=0.3)
            
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        attention_scores = Q @ K.T
        
        scaled_scores = attention_scores / np.sqrt(self.d_model)
        
        attention_weights = softmax(scaled_scores, axis=-1)
        
        fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        im1 = axes[0].imshow(scaled_scores, cmap='RdYlBu', aspect='auto')
        axes[0].set_title('Attention Scores (Escalados)')
        axes[0].set_xlabel('Key Positions')
        axes[0].set_ylabel('Query Positions')
        axes[0].set_xticks(range(self.seq_length))
        axes[0].set_yticks(range(self.seq_length))
        axes[0].set_xticklabels(x_labels, rotation=45, fontsize=9)
        axes[0].set_yticklabels(x_labels, fontsize=9)
        
        for i in range(real_tokens_count, self.seq_length):
            axes[0].axvline(x=i-0.5, color='gray', linestyle='--', alpha=0.3)
            axes[0].axhline(y=i-0.5, color='gray', linestyle='--', alpha=0.3)
            
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(attention_weights, cmap='YlOrRd', aspect='auto')
        axes[1].set_title('Attention Weights (após Softmax)')
        axes[1].set_xlabel('Key Positions')
        axes[1].set_ylabel('Query Positions')
        axes[1].set_xticks(range(self.seq_length))
        axes[1].set_yticks(range(self.seq_length))
        axes[1].set_xticklabels(x_labels, rotation=45, fontsize=9)
        axes[1].set_yticklabels(x_labels, fontsize=9)
        
        for i in range(real_tokens_count, self.seq_length):
            axes[1].axvline(x=i-0.5, color='gray', linestyle='--', alpha=0.3)
            axes[1].axhline(y=i-0.5, color='gray', linestyle='--', alpha=0.3)
        
        for i in range(real_tokens_count):
            for j in range(real_tokens_count):
                text = axes[1].text(j, i, f'{attention_weights[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im2, ax=axes[1])
        plt.tight_layout()
        
        output = attention_weights @ V
        
        fig3, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        im1 = axes[0].imshow(V.T, cmap='Blues', aspect='auto')
        axes[0].set_title('Values Matrix (V)')
        axes[0].set_xlabel('Tokens')
        axes[0].set_ylabel('Dimensões')
        axes[0].set_xticks(range(self.seq_length))
        axes[0].set_xticklabels(x_labels, rotation=45, fontsize=9)
        
        for i in range(real_tokens_count, self.seq_length):
            axes[0].axvline(x=i-0.5, color='gray', linestyle='--', alpha=0.3)
            
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(output.T, cmap='Purples', aspect='auto')
        axes[1].set_title('Attention Output')
        axes[1].set_xlabel('Tokens')
        axes[1].set_ylabel('Dimensões')
        axes[1].set_xticks(range(self.seq_length))
        axes[1].set_xticklabels(x_labels, rotation=45, fontsize=9)
        
        for i in range(real_tokens_count, self.seq_length):
            axes[1].axvline(x=i-0.5, color='gray', linestyle='--', alpha=0.3)
            
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        return fig1, fig2, fig3, Q, K, V, attention_weights, output
    
    def visualize_attention_flow(self, attention_weights, tokens, token_index=3, real_tokens_count=None):
        """Visualiza o fluxo de atenção para um token específico"""
        # Garantir que o índice está dentro dos limites dos tokens reais
        if real_tokens_count is None:
            real_tokens_count = len([t for t in tokens if t])
            
        token_index = min(token_index, real_tokens_count-1)
        
        token_attention = attention_weights[token_index, :]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        token_positions = np.arange(real_tokens_count)
        
        bar_heights = token_attention[:real_tokens_count]
        
        bars = ax.bar(token_positions, bar_heights, color='skyblue', alpha=0.7)
        
        bars[token_index].set_color('red')
        bars[token_index].set_alpha(1.0)
        
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
        
        for i, v in enumerate(bar_heights):
            ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(token_positions)
        ax.set_xticklabels([t for t in tokens[:real_tokens_count]], rotation=45)
        ax.set_ylabel('Peso de Atenção')
        ax.set_title(f'Fluxo de Atenção para o Token "{tokens[token_index]}"')
        
        ax.text(0.5, 0.95, 
                f'O token "{tokens[token_index]}" analisa todos os tokens anteriores\npara construir sua representação contextualizada',
                transform=ax.transAxes, 
                ha='center', 
                va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        
        return fig
    
    def compare_token_importance(self, attention_weights1, tokens1, real_tokens_count1,
                                attention_weights2, tokens2, real_tokens_count2):
        """Compara a importância de tokens entre duas frases"""
        # Calcular importâncias para ambas as frases
        imp_rec1, imp_giv1, imp_comb1 = calculate_token_importance(attention_weights1, tokens1, real_tokens_count1)
        imp_rec2, imp_giv2, imp_comb2 = calculate_token_importance(attention_weights2, tokens2, real_tokens_count2)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        real_tokens1 = tokens1[:real_tokens_count1]
        real_tokens2 = tokens2[:real_tokens_count2]
        
        # Gráfico 1: Importância Recebida (quanto outros tokens prestam atenção)
        x1 = np.arange(len(real_tokens1))
        x2 = np.arange(len(real_tokens2))
        
        bars1 = axes[0,0].bar(x1, imp_rec1, alpha=0.7, color='lightblue', label='Frase 1')
        axes[0,0].set_title('Importância Recebida (Attention IN)')
        axes[0,0].set_xlabel('Tokens')
        axes[0,0].set_ylabel('Soma dos Pesos de Atenção Recebidos')
        axes[0,0].set_xticks(x1)
        axes[0,0].set_xticklabels(real_tokens1, rotation=45)
        
        for i, v in enumerate(imp_rec1):
            axes[0,0].text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        
        bars2 = axes[0,1].bar(x2, imp_rec2, alpha=0.7, color='lightcoral', label='Frase 2')
        axes[0,1].set_title('Importância Recebida (Attention IN)')
        axes[0,1].set_xlabel('Tokens')
        axes[0,1].set_ylabel('Soma dos Pesos de Atenção Recebidos')
        axes[0,1].set_xticks(x2)
        axes[0,1].set_xticklabels(real_tokens2, rotation=45)
        
        for i, v in enumerate(imp_rec2):
            axes[0,1].text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        
        bars3 = axes[1,0].bar(x1, imp_giv1, alpha=0.7, color='lightgreen')
        axes[1,0].set_title('Importância Dada (Attention OUT)')
        axes[1,0].set_xlabel('Tokens')
        axes[1,0].set_ylabel('Soma dos Pesos de Atenção Dados')
        axes[1,0].set_xticks(x1)
        axes[1,0].set_xticklabels(real_tokens1, rotation=45)
        
        for i, v in enumerate(imp_giv1):
            axes[1,0].text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        
        bars4 = axes[1,1].bar(x2, imp_giv2, alpha=0.7, color='gold')
        axes[1,1].set_title('Importância Dada (Attention OUT)')
        axes[1,1].set_xlabel('Tokens')
        axes[1,1].set_ylabel('Soma dos Pesos de Atenção Dados')
        axes[1,1].set_xticks(x2)
        axes[1,1].set_xticklabels(real_tokens2, rotation=45)
        
        for i, v in enumerate(imp_giv2):
            axes[1,1].text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        return fig, (imp_rec1, imp_giv1, imp_comb1), (imp_rec2, imp_giv2, imp_comb2)
    
    def analyze_attention_patterns(self, attention_weights, tokens, real_tokens_count=None):
        """Analisa padrões específicos de atenção"""
        # Determinar o número de tokens reais se não fornecido
        if real_tokens_count is None:
            real_tokens_count = len([t for t in tokens if t])
            
        real_tokens = tokens[:real_tokens_count]
        real_weights = attention_weights[:real_tokens_count, :real_tokens_count]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        token_pos1 = min(1, real_tokens_count-1)
        token1_attention = real_weights[token_pos1, :]
        axes[0,0].bar(range(real_tokens_count), token1_attention, color='skyblue', alpha=0.7)
        axes[0,0].set_title(f'Atenção de "{real_tokens[token_pos1]}" para outros tokens')
        axes[0,0].set_xlabel('Tokens')
        axes[0,0].set_ylabel('Peso de Atenção')
        axes[0,0].set_xticks(range(real_tokens_count))
        axes[0,0].set_xticklabels(real_tokens, rotation=45)
        
        for i, v in enumerate(token1_attention):
            axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        token_pos2 = min(2, real_tokens_count-1)
        token2_attention = real_weights[token_pos2, :]
        axes[0,1].bar(range(real_tokens_count), token2_attention, color='lightcoral', alpha=0.7)
        axes[0,1].set_title(f'Atenção de "{real_tokens[token_pos2]}" para outros tokens')
        axes[0,1].set_xlabel('Tokens')
        axes[0,1].set_ylabel('Peso de Atenção')
        axes[0,1].set_xticks(range(real_tokens_count))
        axes[0,1].set_xticklabels(real_tokens, rotation=45)
        
        for i, v in enumerate(token2_attention):
            axes[0,1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        sns.heatmap(real_weights, annot=True, fmt='.2f', 
                   xticklabels=real_tokens, yticklabels=real_tokens,
                   cmap='YlOrRd', ax=axes[1,0])
        axes[1,0].set_title('Matriz de Atenção Completa')
        axes[1,0].set_xlabel('Attending to (Keys)')
        axes[1,0].set_ylabel('Attending from (Queries)')
        
        axes[1,1].hist(real_weights.flatten(), bins=20, alpha=0.7, color='green')
        axes[1,1].set_title('Distribuição dos Pesos de Atenção')
        axes[1,1].set_xlabel('Valor do Peso')
        axes[1,1].set_ylabel('Frequência')
        axes[1,1].axvline(real_weights.mean(), color='red', linestyle='--', 
                         label=f'Média: {real_weights.mean():.3f}')
        axes[1,1].legend()
        
        plt.tight_layout()
        
        return fig
    
    def compare_attention_patterns(self, attention_weights1, tokens1, real_tokens_count1, 
                                  attention_weights2, tokens2, real_tokens_count2):
        """Compara padrões de atenção entre duas frases"""
        # Usar apenas tokens reais para análise
        real_tokens1 = tokens1[:real_tokens_count1]
        real_weights1 = attention_weights1[:real_tokens_count1, :real_tokens_count1]
        
        real_tokens2 = tokens2[:real_tokens_count2]
        real_weights2 = attention_weights2[:real_tokens_count2, :real_tokens_count2]
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        sns.heatmap(real_weights1, annot=True, fmt='.2f', 
                   xticklabels=real_tokens1, yticklabels=real_tokens1,
                   cmap='YlOrRd', ax=axes[0,0])
        axes[0,0].set_title('Matriz de Atenção - Frase 1')
        axes[0,0].set_xlabel('Attending to (Keys)')
        axes[0,0].set_ylabel('Attending from (Queries)')
        
        sns.heatmap(real_weights2, annot=True, fmt='.2f', 
                   xticklabels=real_tokens2, yticklabels=real_tokens2,
                   cmap='YlOrRd', ax=axes[0,1])
        axes[0,1].set_title('Matriz de Atenção - Frase 2')
        axes[0,1].set_xlabel('Attending to (Keys)')
        axes[0,1].set_ylabel('Attending from (Queries)')
        
        axes[1,0].hist(real_weights1.flatten(), bins=20, alpha=0.7, color='blue', label='Frase 1')
        axes[1,0].set_title('Distribuição dos Pesos - Frase 1')
        axes[1,0].set_xlabel('Valor do Peso')
        axes[1,0].set_ylabel('Frequência')
        axes[1,0].axvline(real_weights1.mean(), color='darkblue', linestyle='--', 
                         label=f'Média: {real_weights1.mean():.3f}')
        axes[1,0].legend()
        
        axes[1,1].hist(real_weights2.flatten(), bins=20, alpha=0.7, color='red', label='Frase 2')
        axes[1,1].set_title('Distribuição dos Pesos - Frase 2')
        axes[1,1].set_xlabel('Valor do Peso')
        axes[1,1].set_ylabel('Frequência')
        axes[1,1].axvline(real_weights2.mean(), color='darkred', linestyle='--', 
                         label=f'Média: {real_weights2.mean():.3f}')
        axes[1,1].legend()
        
        plt.tight_layout()
        
        return fig

class MultiHeadAttention:
    def __init__(self, d_model=64, num_heads=8, seq_length=8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Garantir que d_model é divisível por num_heads
        self.seq_length = seq_length
        
        np.random.seed(42)
        self.W_q_heads = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(num_heads)]
        self.W_k_heads = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(num_heads)]
        self.W_v_heads = [np.random.randn(d_model, self.d_k) * 0.1 for _ in range(num_heads)]
        
        concat_dim = num_heads * self.d_k  # Dimensão após concatenar todas as cabeças
        self.W_o = np.random.randn(concat_dim, d_model) * 0.1
    
    def single_head_attention(self, X, W_q, W_k, W_v):
        """Computa atenção para uma única cabeça"""
        Q = X @ W_q
        K = X @ W_k
        V = X @ W_v
        
        attention_scores = Q @ K.T / np.sqrt(self.d_k)
        attention_weights = softmax(attention_scores, axis=-1)
        output = attention_weights @ V
        
        return output, attention_weights
    
    def compute_multi_head_attention(self, X, tokens, real_tokens_count=None):
        """Computa multi-head attention"""
        # Determinar o número de tokens reais se não fornecido
        if real_tokens_count is None:
            real_tokens_count = len([t for t in tokens if t])
            
        head_outputs = []
        head_attentions = []
        
        for i in range(self.num_heads):
            output, attention = self.single_head_attention(
                X, self.W_q_heads[i], self.W_k_heads[i], self.W_v_heads[i]
            )
            head_outputs.append(output)
            head_attentions.append(attention)
        
        concatenated = np.concatenate(head_outputs, axis=-1)
        
        concat_dim = concatenated.shape[-1]
        expected_dim = self.num_heads * self.d_k
        
        if concat_dim != expected_dim:
            st.warning(f"⚠️ Dimensão inesperada: {concat_dim} vs {expected_dim}")
            # Ajustar W_o se necessário
            if self.W_o.shape[0] != concat_dim:
                self.W_o = np.random.randn(concat_dim, self.d_model) * 0.1
        
        final_output = concatenated @ self.W_o
        
        fig = self.visualize_multi_head_patterns(head_attentions, tokens, real_tokens_count)
        
        return fig, final_output, head_attentions
    
    def visualize_multi_head_patterns(self, head_attentions, tokens, real_tokens_count=None):
        """Visualiza padrões de atenção de diferentes cabeças"""
        # Determinar o número de tokens reais se não fornecido
        if real_tokens_count is None:
            real_tokens_count = len([t for t in tokens if t])
            
        real_tokens = tokens[:real_tokens_count]
        
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
                # Extrair apenas a parte relevante da matriz de atenção (tokens reais)
                attention_display = head_attentions[i][:real_tokens_count, :real_tokens_count]
                
                im = axes[i].imshow(attention_display, cmap='YlOrRd', aspect='auto')
                axes[i].set_title(f'Cabeça {i+1}')
                axes[i].set_xticks(range(real_tokens_count))
                axes[i].set_yticks(range(real_tokens_count))
                axes[i].set_xticklabels(real_tokens, rotation=45, fontsize=8)
                axes[i].set_yticklabels(real_tokens, fontsize=8)
                plt.colorbar(im, ax=axes[i])
        
        for i in range(self.num_heads, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Padrões de Atenção por Cabeça', fontsize=16)
        plt.tight_layout()
        
        return fig

def main():
    # Introdução com explicação sobre LLMs
    st.markdown("""
    <div class="highlight">
        <h2>Entendendo o Mecanismo de Attention em Transformers</h2>
        <p>Esta aplicação demonstra visualmente como funciona o mecanismo de Attention, 
        componente fundamental das arquiteturas Transformer que revolucionaram o Processamento 
        de Linguagem Natural e outras áreas de IA.</p>
        <p>Agora com análise de importância de tokens e comparação detalhada entre frases!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("🧠 Como Funcionam os Large Language Models (LLMs)")
    
    st.markdown("""
    <div class="llm-explanation">
        <h3>Fundamentos dos Modelos de Linguagem</h3>
        <p><strong>Large Language Models (LLMs)</strong> como GPT, BERT e outros são redes neurais gigantescas treinadas em vastos conjuntos de texto para entender e gerar linguagem humana. Eles funcionam através de:</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 1. Tokenização e Embeddings")
    st.markdown("""
    - **Tokenização:** O texto é dividido em unidades menores (tokens) - palavras, subpalavras ou caracteres
    - **Embeddings:** Cada token é convertido em um vetor numérico denso que captura seu significado semântico
    - **Positional Encoding:** Como os Transformers processam todos os tokens simultaneamente, precisamos adicionar informação sobre a posição de cada palavra
    """)
    
    st.markdown("#### 2. Mecanismo de Attention")
    st.markdown("""
    - **Self-Attention:** Cada token "presta atenção" a todos os outros tokens da sequência
    - **Context Understanding:** Isso permite que o modelo entenda como palavras se relacionam, mesmo estando distantes na frase
    - **Múltiplas Cabeças:** Diferentes "cabeças de atenção" capturam diferentes tipos de relações (sintáticas, semânticas, etc.)
    """)
    
    st.markdown("#### 3. Processamento em Camadas")
    st.markdown("""
    - **Múltiplas Camadas:** Os LLMs têm dezenas ou centenas de camadas Transformer empilhadas
    - **Representações Hierárquicas:** Cada camada constrói representações mais complexas baseadas na anterior
    - **Emergência:** Comportamentos complexos emergem da interação entre essas camadas simples
    """)
    
    st.markdown("#### 4. Treinamento e Previsão")
    st.markdown("""
    - **Previsão de Próxima Palavra:** Durante o treinamento, o modelo aprende a prever a próxima palavra em uma sequência
    - **Aprendizado de Padrões:** Isso força o modelo a aprender gramática, semântica, fatos sobre o mundo, e muito mais
    - **Transferência:** Uma vez treinado, o modelo pode ser adaptado para diversas tarefas específicas
    """)
    
    st.markdown("""
    <div class="llm-explanation">
        <p><strong>O resultado:</strong> Um modelo capaz de entender contexto, gerar texto coerente, responder perguntas, traduzir idiomas e muito mais!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Seção para geração de frases com OpenAI
    st.header("📝 Geração de Frases para Comparação")
    
    st.markdown("""
    <div class="explanation">
        <p>Utilizaremos duas frases para demonstrar como o mecanismo de atenção processa diferentes contextos. 
        Você pode usar frases de exemplo ou gerar novas frases com palavras similares usando a API OpenAI.</p>
    </div>
    """, unsafe_allow_html=True)
    
    sentence1 = "O gato de maria caminha devagar pela floresta verde escura."
    sentence2 = "A galinha de fazenda bota ovos frescos no galinheiro."
    
    use_example = st.checkbox("Usar frases de exemplo (sem API OpenAI)", value=True)
    
    if not use_example and api_key:
        # Seção para geração com OpenAI
        st.markdown("### 🤖 Geração com OpenAI")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Configure sua chave da API no painel lateral e clique no botão para gerar frases personalizadas.")
        with col2:
            generate_button = st.button("🎲 Gerar Frases", type="primary")
        
        if generate_button:
            try:
                with st.spinner("🔄 Gerando frases com a API OpenAI..."):
                    prompt = """
                    Gere duas frases em português que atendam aos seguintes critérios:
                    
                    1. Cada frase deve ter EXATAMENTE 10 palavras (incluindo artigos, preposições, etc.)
                    2. As frases devem compartilhar pelo menos 2-3 palavras similares ou relacionadas
                    3. As palavras similares devem ter significados ou usos diferentes nos dois contextos
                    4. Use vocabulário simples e claro
                    5. Evite pontuação complexa
                    
                    Exemplos do que procuro:
                    - Palavras como "banco" (móvel vs instituição financeira)
                    - Palavras como "manga" (fruta vs parte da roupa)
                    - Palavras como "casa" em contextos diferentes
                    
                    Forneça apenas as duas frases, uma por linha, sem numeração ou explicações.
                    """
                    
                    result = generate_sentences_with_openai(api_key, prompt)
                    
                    if result:
                        # Dividir o resultado em duas frases
                        sentences = [s.strip().rstrip('.').rstrip('!').rstrip('?') for s in result.split('\n') if s.strip()]
                        
                        if len(sentences) >= 2:
                            sentence1 = sentences[0]
                            sentence2 = sentences[1]
                            
                            # Garantir que termine com ponto
                            if not sentence1.endswith('.'):
                                sentence1 += "."
                            if not sentence2.endswith('.'):
                                sentence2 += "."
                            
                            st.session_state['generated_sentence1'] = sentence1
                            st.session_state['generated_sentence2'] = sentence2
                            
                            st.success("✅ Frases geradas com sucesso!")
                            st.rerun()
                        else:
                            st.error("❌ Não foi possível obter duas frases distintas da API.")
                    
            except Exception as e:
                st.error(f"❌ Erro ao gerar frases: {str(e)}")
                st.info("💡 Verifique se sua chave da API está correta e tente novamente.")
        
        if 'generated_sentence1' in st.session_state and 'generated_sentence2' in st.session_state:
            sentence1 = st.session_state['generated_sentence1']
            sentence2 = st.session_state['generated_sentence2']
            
            if st.button("🔄 Usar Frases de Exemplo"):
                del st.session_state['generated_sentence1']
                del st.session_state['generated_sentence2']
                st.rerun()
    
    elif not use_example and not api_key:
        st.warning("⚠️ Insira sua chave da API OpenAI no painel lateral para gerar frases personalizadas.")
        st.info("📝 Usando frases de exemplo por enquanto...")
    
    st.markdown(f"""
    <div class="comparison-container">
        <div class="comparison-card">
            <h4>Frase 1 ({len(sentence1.split())} palavras):</h4>
            <p>{sentence1}</p>
        </div>
        <div class="comparison-card">
            <h4>Frase 2 ({len(sentence2.split())} palavras):</h4>
            <p>{sentence2}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tokens1 = tokenize_sentence(sentence1)
    tokens2 = tokenize_sentence(sentence2)
    
    simulator = TransformerSimulator(d_model=d_model, seq_length=seq_length)
    
    st.markdown("---")
    
    st.header("1. 🔢 Embeddings e Positional Encoding")
    
    st.markdown("""
    <div class="explanation">
        <p>Em um modelo Transformer, as palavras (tokens) são primeiro convertidas em vetores densos chamados <b>embeddings</b>. 
        Como os Transformers processam todos os tokens simultaneamente (e não sequencialmente como RNNs), 
        precisamos adicionar informação sobre a posição de cada token na sequência. Isso é feito através do <b>Positional Encoding</b>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Frase 1", "Frase 2"])
    
    with tab1:
        fig1, embeddings1, tokens_used1, real_tokens_count1 = simulator.visualize_embeddings_and_positional(tokens1)
        st.pyplot(fig1)
    
    with tab2:
        fig2, embeddings2, tokens_used2, real_tokens_count2 = simulator.visualize_embeddings_and_positional(tokens2)
        st.pyplot(fig2)
    
    st.markdown("""
    <div class="explanation">
        <p><b>Explicação dos gráficos:</b></p>
        <ol>
            <li><b>Token Embeddings</b>: Representação vetorial de cada palavra aprendida durante o treinamento.</li>
            <li><b>Positional Encoding</b>: Informação sobre a posição usando funções seno/cosseno para preservar ordem.</li>
            <li><b>Embeddings Finais</b>: Combinação que permite ao modelo saber "o que" é cada palavra e "onde" ela está.</li>
            <li><b>Padrões Sinusoidais</b>: As diferentes frequências permitem ao modelo distinguir posições próximas e distantes.</li>
        </ol>
        <p>💡 <b>Dica</b>: As linhas tracejadas separam tokens reais dos tokens de padding (preenchimento).</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header("2. 🔍 Mecanismo de Self-Attention Passo a Passo")
    
    st.markdown("""
    <div class="explanation">
        <p>O <b>Self-Attention</b> é o coração dos Transformers. Ele permite que cada palavra "converse" com todas as outras palavras da frase, 
        capturando relações complexas independentemente da distância. Vamos ver cada passo:</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Frase 1", "Frase 2"])
    
    with tab1:
        st.subheader("🔍 Passo 1: Criando Query, Key e Value matrices")
        
        st.markdown("""
        <div class="explanation">
            <p>Para cada token, criamos três representações diferentes:</p>
            <ul>
                <li><b>Query (Q)</b>: "O que este token está procurando?" - determina que tipo de informação é relevante</li>
                <li><b>Key (K)</b>: "Que informação este token oferece?" - representa o conteúdo disponível</li>
                <li><b>Value (V)</b>: "Qual informação será realmente transmitida?" - o conteúdo efetivo a ser agregado</li>
            </ul>
            <p>🔧 <b>Implementação</b>: Multiplicamos os embeddings por matrizes de peso treináveis (W_q, W_k, W_v).</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig_qkv1, fig_scores1, fig_output1, Q1, K1, V1, attention_weights1, output1 = simulator.compute_attention_step_by_step(embeddings1, tokens_used1, real_tokens_count1)
        st.pyplot(fig_qkv1)
        
        st.subheader("🧮 Passo 2 & 3: Calculando Attention Scores e Aplicando Softmax")
        
        st.markdown(f"""
        <div class="explanation">
            <p><b>Passo 2:</b> Calculamos a "compatibilidade" entre cada Query e cada Key através do produto escalar Q·K<sup>T</sup></p>
            <p>📏 <b>Escalamento</b>: Dividimos por √{d_model} = {1/np.sqrt(d_model):.3f} para estabilizar os gradientes</p>
            <p><b>Passo 3:</b> Aplicamos softmax para converter scores em probabilidades que somam 1</p>
            <p>💡 <b>Intuição</b>: Quanto maior o score, mais "atenção" um token dará ao outro</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.pyplot(fig_scores1)
        
        st.subheader("🎯 Passo 4: Computando Output (Attention × Values)")
        
        st.markdown("""
        <div class="explanation">
            <p>Finalmente, usamos os pesos de atenção para fazer uma média ponderada dos Values.</p>
            <p>🎭 <b>Resultado</b>: Cada token agora contém informação contextualizada de toda a sequência!</p>
            <p>✨ <b>Magia</b>: O modelo aprendeu automaticamente quais palavras são importantes para cada contexto</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.pyplot(fig_output1)
        
        st.subheader("🔄 Fluxo de Atenção para Tokens Específicos")
        
        st.markdown("""
        <div class="attention-flow">
            <p><b>🧠 Como o modelo "pensa":</b> Cada token analisa todos os tokens anteriores para construir sua representação. 
            As setas mostram as conexões mais fortes - onde o modelo está "prestando mais atenção".</p>
            <p>🔍 <b>Experimente</b>: Mude o token selecionado para ver como diferentes palavras focam em aspectos diferentes da frase!</p>
        </div>
        """, unsafe_allow_html=True)
        
        token_to_analyze1 = st.slider("Selecione o token para analisar (Frase 1)", 
                                     min_value=2, 
                                     max_value=min(real_tokens_count1-1, 9), 
                                     value=min(3, real_tokens_count1-1),
                                     help="Escolha um token para visualizar como ele presta atenção aos tokens anteriores")
        
        flow_fig1 = simulator.visualize_attention_flow(attention_weights1, tokens_used1, token_to_analyze1, real_tokens_count1)
        st.pyplot(flow_fig1)
    
    with tab2:
        st.subheader("🔍 Passo 1: Criando Query, Key e Value matrices")
        
        st.markdown("""
        <div class="explanation">
            <p>Observe como os mesmos passos aplicados à segunda frase produzem padrões diferentes:</p>
            <ul>
                <li><b>Query (Q)</b>: Cada token busca informações relevantes no contexto da segunda frase</li>
                <li><b>Key (K)</b>: As "chaves" que cada token oferece dependem do vocabulário e contexto</li>
                <li><b>Value (V)</b>: O conteúdo semântico varia conforme a estrutura da frase</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        fig_qkv2, fig_scores2, fig_output2, Q2, K2, V2, attention_weights2, output2 = simulator.compute_attention_step_by_step(embeddings2, tokens_used2, real_tokens_count2)
        st.pyplot(fig_qkv2)
        
        st.subheader("🧮 Passo 2 & 3: Calculando Attention Scores e Aplicando Softmax")
        
        st.markdown(f"""
        <div class="explanation">
            <p><b>Compare:</b> Note como os padrões de scores diferem da primeira frase</p>
            <p>🎯 <b>Insight</b>: Palavras similares em contextos diferentes geram scores únicos</p>
            <p>📊 <b>Softmax</b>: Normaliza os scores para que cada linha some exatamente 1.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.pyplot(fig_scores2)
        
        st.subheader("🎯 Passo 4: Computando Output (Attention × Values)")
        
        st.markdown("""
        <div class="explanation">
            <p>🔄 <b>Agregação contextual</b>: Cada posição recebe uma mistura ponderada de informações</p>
            <p>🌟 <b>Emergência</b>: O significado final emerge da interação entre todos os tokens</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.pyplot(fig_output2)
        
        st.subheader("🔄 Fluxo de Atenção para Tokens Específicos")
        
        st.markdown("""
        <div class="attention-flow">
            <p><b>🔬 Análise comparativa:</b> Compare os padrões de atenção entre as duas frases. 
            Palavras em posições similares podem ter comportamentos muito diferentes!</p>
        </div>
        """, unsafe_allow_html=True)
        
        token_to_analyze2 = st.slider("Selecione o token para analisar (Frase 2)", 
                                     min_value=2, 
                                     max_value=min(real_tokens_count2-1, 9), 
                                     value=min(3, real_tokens_count2-1),
                                     help="Escolha um token para visualizar como ele presta atenção aos tokens anteriores")
        
        flow_fig2 = simulator.visualize_attention_flow(attention_weights2, tokens_used2, token_to_analyze2, real_tokens_count2)
        st.pyplot(flow_fig2)
    
    st.markdown("---")
    
    st.header("3. ⚖️ Análise de Importância de Tokens")
    
    st.markdown("""
    <div class="importance-analysis">
        <h3>🎯 Como Medimos a Importância de um Token?</h3>
        <p>Analisamos dois aspectos fundamentais do comportamento de atenção:</p>
        <ul>
            <li><b>Importância Recebida (Attention IN)</b>: Quanto outros tokens prestam atenção a este token
                <br>➡️ <i>Indica quão "central" ou "importante" uma palavra é para o contexto geral</i></li>
            <li><b>Importância Dada (Attention OUT)</b>: Quanto este token presta atenção aos outros
                <br>➡️ <i>Indica quão "ativo" um token é em buscar informações contextuais</i></li>
        </ul>
        <p>🧮 <b>Cálculo</b>: Somamos os pesos de atenção recebidos/dados por cada token na matriz de atenção</p>
    </div>
    """, unsafe_allow_html=True)
    
    fig_importance, imp_data1, imp_data2 = simulator.compare_token_importance(
        attention_weights1, tokens_used1, real_tokens_count1,
        attention_weights2, tokens_used2, real_tokens_count2
    )
    
    st.pyplot(fig_importance)
    
    st.subheader("🔍 Comparação Token por Token")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Frase 1: Ranking de Importância**")
        # Criar dataframe para frase 1
        real_tokens1 = tokens_used1[:real_tokens_count1]
        imp_rec1, imp_giv1, imp_comb1 = imp_data1
        
        df1 = pd.DataFrame({
            'Token': real_tokens1,
            'Posição': range(len(real_tokens1)),
            'Atenção Recebida': imp_rec1,
            'Atenção Dada': imp_giv1,
            'Importância Combinada': imp_comb1
        }).round(3)
        
        df1_sorted = df1.sort_values('Importância Combinada', ascending=False)
        st.dataframe(df1_sorted, use_container_width=True)
        
        # Destacar top 3
        top3_1 = df1_sorted.head(3)['Token'].tolist()
        st.markdown(f"🏆 **Top 3 mais importantes:** {', '.join(top3_1)}")
    
    with col2:
        st.markdown("**📊 Frase 2: Ranking de Importância**")
        # Criar dataframe para frase 2
        real_tokens2 = tokens_used2[:real_tokens_count2]
        imp_rec2, imp_giv2, imp_comb2 = imp_data2
        
        df2 = pd.DataFrame({
            'Token': real_tokens2,
            'Posição': range(len(real_tokens2)),
            'Atenção Recebida': imp_rec2,
            'Atenção Dada': imp_giv2,
            'Importância Combinada': imp_comb2
        }).round(3)
        
        df2_sorted = df2.sort_values('Importância Combinada', ascending=False)
        st.dataframe(df2_sorted, use_container_width=True)
        
        top3_2 = df2_sorted.head(3)['Token'].tolist()
        st.markdown(f"🏆 **Top 3 mais importantes:** {', '.join(top3_2)}")
    
    st.subheader("🔄 Análise de Palavras Similares")
    
    set1 = set(real_tokens1)
    set2 = set(real_tokens2)
    common_words = set1.intersection(set2)
    
    if common_words:
        st.markdown(f"**🔗 Palavras em comum encontradas:** {', '.join(common_words)}")
        
        for word in common_words:
            if word in real_tokens1 and word in real_tokens2:
                pos1 = real_tokens1.index(word)
                pos2 = real_tokens2.index(word)
                
                imp1 = imp_comb1[pos1]
                imp2 = imp_comb2[pos2]
                
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    st.metric(f"'{word}' - Frase 1", f"{imp1:.3f}", f"Posição {pos1}")
                with col2:
                    st.metric(f"'{word}' - Frase 2", f"{imp2:.3f}", f"Posição {pos2}")
                with col3:
                    diff = imp2 - imp1
                    direction = "maior" if diff > 0 else "menor"
                    st.markdown(f"**Diferença:** {abs(diff):.3f}")
                    st.markdown(f"A palavra '{word}' tem importância {direction} na Frase 2")
    else:
        st.markdown("**ℹ️ Nenhuma palavra exatamente igual encontrada entre as frases.**")
        
        # Buscar palavras similares (mesmo começo)
        similar_pairs = []
        for w1 in real_tokens1:
            for w2 in real_tokens2:
                if len(w1) > 2 and len(w2) > 2 and w1[:3] == w2[:3] and w1 != w2:
                    similar_pairs.append((w1, w2))
        
        if similar_pairs:
            st.markdown(f"**🔗 Palavras similares encontradas:** {', '.join([f'{w1}↔{w2}' for w1, w2 in similar_pairs[:3]])}")
    
    st.markdown("---")
    
    st.header("4. 📈 Análise Comparativa de Padrões de Atenção")
    
    st.markdown("""
    <div class="explanation">
        <p>Agora vamos comparar os padrões emergentes de atenção entre as duas frases. Esta análise revela como 
        o contexto influencia fundamentalmente o processamento de cada palavra.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Análise Individual", "Comparação Direta"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Frase 1 - Padrões Detalhados")
            fig_patterns1 = simulator.analyze_attention_patterns(attention_weights1, tokens_used1, real_tokens_count1)
            st.pyplot(fig_patterns1)
        
        with col2:
            st.subheader("📊 Frase 2 - Padrões Detalhados")
            fig_patterns2 = simulator.analyze_attention_patterns(attention_weights2, tokens_used2, real_tokens_count2)
            st.pyplot(fig_patterns2)
    
    with tab2:
        st.subheader("🔄 Comparação Direta dos Padrões de Atenção")
        fig_comparison = simulator.compare_attention_patterns(
            attention_weights1, tokens_used1, real_tokens_count1, 
            attention_weights2, tokens_used2, real_tokens_count2
        )
        st.pyplot(fig_comparison)
        
        st.markdown("""
        <div class="explanation">
            <h4>🧠 Insights sobre as diferenças:</h4>
            <ul>
                <li><strong>📍 Contexto é rei:</strong> Palavras similares em contextos diferentes apresentam padrões de atenção únicos</li>
                <li><strong>📊 Distribuição de pesos:</strong> A "forma" da distribuição revela a complexidade sintática da frase</li>
                <li><strong>🎭 Papéis sintáticos:</strong> Substantivos, verbos e modificadores mostram comportamentos característicos</li>
                <li><strong>🔗 Dependências:</strong> Palavras funcionais (artigos, preposições) tendem a ter padrões mais dispersos</li>
                <li><strong>⚡ Emergência:</strong> Padrões complexos emergem automaticamente do treinamento simples</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Parte 5: Multi-Head Attention
    st.header("5. 🧠 Multi-Head Attention")
    
    st.markdown("""
    <div class="explanation">
        <p>Em vez de ter apenas um mecanismo de atenção, os Transformers usam <b>múltiplas cabeças de atenção</b> em paralelo.
        Cada cabeça pode se especializar em diferentes aspectos da linguagem:</p>
        <ul>
            <li>🔗 <b>Relações sintáticas</b> (sujeito-verbo, modificador-substantivo)</li>
            <li>🎭 <b>Relações semânticas</b> (sinônimos, antonimos, categorias)</li>
            <li>📍 <b>Proximidade posicional</b> (palavras próximas vs distantes)</li>
            <li>🎯 <b>Referências</b> (pronomes, anáforas)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, seq_length=seq_length)
    
    tab1, tab2 = st.tabs(["Frase 1", "Frase 2"])
    
    with tab1:
        fig_mha1, final_output1, head_attentions1 = mha.compute_multi_head_attention(
            embeddings1, tokens_used1, real_tokens_count1
        )
        
        st.markdown(f"""
        <div class="explanation">
            <p><strong>⚙️ Configuração atual:</strong></p>
            <ul>
                <li>🧠 Número de cabeças: <strong>{num_heads}</strong></li>
                <li>📏 Dimensão por cabeça (d_k): <strong>{d_model // num_heads}</strong></li>
                <li>🎯 Dimensão total do modelo (d_model): <strong>{d_model}</strong></li>
                <li>🔗 Dimensão após concatenação: <strong>{num_heads * (d_model // num_heads)}</strong></li>
            </ul>
            <p>💡 <strong>Observe:</strong> Cada cabeça captura padrões únicos - algumas focam em posições próximas, outras em relações específicas!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.pyplot(fig_mha1)
    
    with tab2:
        fig_mha2, final_output2, head_attentions2 = mha.compute_multi_head_attention(
            embeddings2, tokens_used2, real_tokens_count2
        )
        
        st.markdown(f"""
        <div class="explanation">
            <p><strong>🔬 Análise comparativa:</strong> Compare como as mesmas {num_heads} cabeças se comportam diferentemente 
            na segunda frase. Isso demonstra a adaptabilidade do mecanismo de atenção!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.pyplot(fig_mha2)
    
    st.markdown("""
    <div class="explanation">
        <h4>🎭 Especialização das Cabeças</h4>
        <p>Cada cabeça de atenção desenvolve "personalidades" distintas durante o treinamento:</p>
        <ul>
            <li><strong>🎯 Cabeças focais:</strong> Concentram atenção em poucas palavras específicas</li>
            <li><strong>🌊 Cabeças difusas:</strong> Distribuem atenção mais uniformemente</li>
            <li><strong>📍 Cabeças posicionais:</strong> Focam em proximidade física na sequência</li>
            <li><strong>🔗 Cabeças relacionais:</strong> Capturam dependências sintáticas específicas</li>
        </ul>
        <p><strong>🔄 Combinação final:</strong> Os outputs de todas as cabeças são concatenados e projetados para produzir 
        a representação final, rica em múltiplas perspectivas da mesma sequência!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("🎓 Conclusão e Próximos Passos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="highlight">
            <h3>🧠 O que aprendemos</h3>
            <p>O mecanismo de Attention é revolucionário porque:</p>
            <ul>
                <li><strong>🔄 Processamento paralelo:</strong> Todos os tokens são processados simultaneamente</li>
                <li><strong>🎯 Atenção seletiva:</strong> Cada palavra pode focar nas informações mais relevantes</li>
                <li><strong>📍 Consciência posicional:</strong> O modelo sabe onde cada palavra está na sequência</li>
                <li><strong>🧠 Múltiplas perspectivas:</strong> Diferentes cabeças capturam diferentes aspectos</li>
                <li><strong>⚖️ Importância contextual:</strong> A mesma palavra pode ter importâncias diferentes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="llm-explanation">
            <h3>🚀 Próximos Passos</h3>
            <p>Para aprofundar seu entendimento:</p>
            <ul>
                <li><strong>🎛️ Experimente:</strong> Ajuste o número de cabeças no painel lateral</li>
                <li><strong>📝 Gere:</strong> Teste diferentes tipos de frases com a API OpenAI</li>
                <li><strong>🔍 Analise:</strong> Observe como palavras similares se comportam diferentemente</li>
                <li><strong>📚 Estude:</strong> Explore papers sobre Transformer, BERT, GPT</li>
                <li><strong>💻 Implemente:</strong> Tente programar seu próprio mecanismo de atenção</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
        <h3>🌟 O Impacto dos Transformers</h3>
        <p>Esta arquitetura revolucionou não apenas o NLP, mas toda a IA:</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **🗣️ Processamento de Linguagem:** BERT, GPT, T5, ChatGPT, Claude
    
    **🖼️ Visão Computacional:** Vision Transformer (ViT), DALL-E
    
    **🎵 Áudio:** Whisper, MusicLM
    
    **🧬 Ciências:** AlphaFold, modelos de proteínas
    
    **🤖 IA Geral:** Modelos multimodais como GPT-4V
    """)
    
    st.markdown("""
    <div class="highlight">
        <p><strong>🎯 A chave do sucesso:</strong> A capacidade de capturar relações complexas através de um mecanismo 
        elegante e paralelizável que escala com dados e computação!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
