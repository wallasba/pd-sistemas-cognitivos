import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from pyvis.network import Network
from src.graph_builder import filter_top_nodes_by_criterion
import streamlit.components.v1 as components
import os

def filter_top_nodes(G, top_n=30):
    """Filtra os N nós mais importantes (por grau)."""
    if G is None or G.number_of_nodes() == 0:
        return G
    if G.number_of_nodes() <= top_n:
        return G
    deg = dict(G.degree())
    sorted_nodes = sorted(deg.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in sorted_nodes[:top_n]]
    return G.subgraph(top_nodes).copy()

def prepare_node_attributes(G, metric='degree', scale_factor=1.0, min_size=5, max_size=60):
    """
    Adiciona atributos 'size' e 'color' aos nós.
    - scale_factor: multiplicador global para ajustar tamanhos
    - min_size, max_size: limites mín/máx do tamanho do nó
    """
    if G is None or G.number_of_nodes() == 0:
        return G
    
    G_copy = G.copy()
    
    # Métrica para tamanho
    if metric == 'degree':
        values = dict(G_copy.degree())
    elif metric == 'betweenness':
        try:
            values = nx.betweenness_centrality(G_copy)
        except:
            values = dict(G_copy.degree())
    elif metric == 'closeness':
        try:
            values = nx.closeness_centrality(G_copy)
        except:
            values = dict(G_copy.degree())
    else:  # 'uniform'
        values = {node: 1 for node in G_copy.nodes}
    
    # Normaliza e aplica escala
    max_val = max(values.values()) if values else 1
    min_val = min(values.values()) if values else 0
    range_val = max_val - min_val if max_val != min_val else 1
    
    for node in G_copy.nodes:
        raw = values.get(node, 0)
        # Normaliza entre 0 e 1
        norm = (raw - min_val) / range_val if range_val > 0 else 0.5
        # Aplica escala e limites
        size = min_size + norm * (max_size - min_size)
        size = size * scale_factor
        G_copy.nodes[node]['size'] = max(min_size, min(max_size, size))
    
    # Cor (comunidades ou uniforme)
    if G_copy.number_of_nodes() > 2 and st.session_state.get('color_community', True):
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G_copy, seed=42)
            colors = ['#FF6B6B','#4ECDC4','#45B7D1','#FFA07A','#98D8C8','#DDA0DD','#F0E68C','#FFD700','#87CEEB','#FF69B4']
            color_map = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    color_map[node] = colors[i % len(colors)]
            for node in G_copy.nodes:
                G_copy.nodes[node]['color'] = color_map.get(node, '#D3D3D3')
        except:
            for node in G_copy.nodes:
                G_copy.nodes[node]['color'] = '#4A90D9'
    else:
        for node in G_copy.nodes:
            G_copy.nodes[node]['color'] = '#4A90D9'
    
    return G_copy

def display_graph(
    G, 
    title, 
    key_suffix, 
    view_mode, 
    top_n=30,
    filter_criterion = 'degree',

    # Parâmetros visuais ajustáveis
    node_metric='degree',
    node_scale=1.0,
    node_min_size=10,
    node_max_size=80,
    edge_scale=1.0,
    edge_min_width=0.5,
    edge_max_width=5.0,
    font_size=10,
    fig_width=14,
    fig_height=10,
    layout_type='spring',
    layout_k=0.3,
    layout_iterations=100,
    show_labels=True,
    label_trim=25
):
    """
    Exibe grafo com parâmetros totalmente configuráveis para legibilidade.
    """
    if G is None or G.number_of_nodes() == 0:
        st.info(f"ℹ️ Sem dados para gerar o grafo '{title}'.")
        return
    
    G_filtered = filter_top_nodes_by_criterion(G, top_n, filter_criterion)
    if G_filtered is None or G_filtered.number_of_nodes() < 2:
        st.info(f"ℹ️ Grafo muito pequeno para visualização (menos de 2 nós).")
        return
    
    # Aplica atributos visuais com parâmetros
    G_filtered = prepare_node_attributes(
        G_filtered, 
        metric=node_metric,
        scale_factor=node_scale,
        min_size=node_min_size,
        max_size=node_max_size
    )
    
    total_nodes = G.number_of_nodes()
    displayed_nodes = G_filtered.number_of_nodes()
    st.caption(f"📊 Exibindo {displayed_nodes} nós (dos {total_nodes} totais).")
    
    cache_key = f"graph_{key_suffix}"
    
    # ============================================================
    # MODO INTERATIVO (PyVis)
    # ============================================================
    if view_mode == "Interativo (com zoom e arraste)":
        try:
            if cache_key not in st.session_state or st.session_state.get(f"{cache_key}_force", False):
                net = Network(
                    height=f"{fig_height*60}px", 
                    width="100%", 
                    notebook=False, 
                    bgcolor="#ffffff"
                )
                net.from_nx(G_filtered)
                
                # Aplica tamanho e cor
                for node in G_filtered.nodes:
                    if node in net.nodes:
                        net.nodes[node]['size'] = G_filtered.nodes[node].get('size', 30)
                        net.nodes[node]['color'] = G_filtered.nodes[node].get('color', '#4A90D9')
                        # Título para tooltip
                        if 'title' in G_filtered.nodes[node]:
                            net.nodes[node]['title'] = G_filtered.nodes[node]['title']
                
                # Aplica espessura das arestas
                for edge in G_filtered.edges(data=True):
                    u, v, data = edge
                    weight = data.get('weight', 1)
                    # Aplica escala e limites
                    width = edge_min_width + (weight / (max([d.get('weight',1) for _,_,d in G_filtered.edges(data=True)] or [1]))) * (edge_max_width - edge_min_width)
                    width = width * edge_scale
                    width = max(edge_min_width, min(edge_max_width, width))
                    if net.get_edge(u, v):
                        net.edges[net.get_edge(u, v)]['width'] = width
                
                # Desativa física para estabilidade
                net.set_options("""
                var options = {
                  "physics": { "enabled": false },
                  "interaction": { "hover": true, "tooltipDelay": 200 }
                }
                """)
                
                filename = f"graph_{key_suffix}.html"
                net.write_html(filename)
                with open(filename, "r", encoding="utf-8") as f:
                    html_content = f.read()
                try:
                    os.remove(filename)
                except:
                    pass
                
                if len(html_content) > 1000:
                    st.session_state[cache_key] = html_content
                    st.session_state[f"{cache_key}_force"] = False
                else:
                    raise Exception("HTML muito pequeno.")
            
            if cache_key in st.session_state:
                st.components.v1.html(st.session_state[cache_key], height=fig_height*60+50)
                st.caption("🖱️ Arraste para mover, role para zoom. Física desativada.")
                if st.button("🔄 Recarregar grafo", key=f"reload_{key_suffix}"):
                    st.session_state[f"{cache_key}_force"] = True
                    st.rerun()
            else:
                st.warning("Grafo não disponível.")
        except Exception as e:
            st.warning(f"⚠️ Falha no modo interativo: {e}")
            st.info("🔄 Alternando para modo estático...")
            display_graph(G, title, key_suffix, "Estático (imagem fixa)", top_n,
                         node_metric, node_scale, node_min_size, node_max_size,
                         edge_scale, edge_min_width, edge_max_width,
                         font_size, fig_width, fig_height, layout_type,
                         layout_k, layout_iterations, show_labels, label_trim)
    
    # ============================================================
    # MODO ESTÁTICO (Matplotlib) – TOTALMENTE CONFIGURÁVEL
    # ============================================================
    else:
        try:
            # Extrai atributos
            sizes = []
            colors = []
            labels = {}
            for node in G_filtered.nodes:
                sizes.append(G_filtered.nodes[node].get('size', 30))
                colors.append(G_filtered.nodes[node].get('color', '#4A90D9'))
                label = str(node)
                if len(label) > label_trim:
                    label = label[:label_trim-3] + "..."
                labels[node] = label
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Layout escolhido
            if layout_type == 'spring':
                pos = nx.spring_layout(G_filtered, seed=42, k=layout_k, iterations=layout_iterations)
            elif layout_type == 'circular':
                pos = nx.circular_layout(G_filtered)
            elif layout_type == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G_filtered)
            elif layout_type == 'random':
                pos = nx.random_layout(G_filtered, seed=42)
            else:
                pos = nx.spring_layout(G_filtered, seed=42, k=0.3, iterations=100)
            
            # Desenha arestas com espessura proporcional ao peso
            edges = G_filtered.edges(data=True)
            edge_weights = [data.get('weight', 1) for _, _, data in edges]
            if edge_weights:
                max_w = max(edge_weights) if edge_weights else 1
                # Aplica escala e limites
                edge_widths = []
                for w in edge_weights:
                    width = edge_min_width + (w / max_w) * (edge_max_width - edge_min_width)
                    width = width * edge_scale
                    edge_widths.append(max(edge_min_width, min(edge_max_width, width)))
            else:
                edge_widths = [edge_min_width] * len(edges)
            
            nx.draw_networkx_edges(
                G_filtered, pos, ax=ax,
                alpha=0.4, width=edge_widths, edge_color='gray'
            )
            
            # Desenha nós com tamanho e cor
            nx.draw_networkx_nodes(
                G_filtered, pos, ax=ax,
                node_size=sizes, node_color=colors, alpha=0.85,
                edgecolors='black', linewidths=0.5
            )
            
            # Rótulos (se ativado)
            if show_labels:
                nx.draw_networkx_labels(
                    G_filtered, pos, ax=ax,
                    labels=labels, font_size=font_size, font_weight='bold',
                    font_color='black'
                )
            
            ax.set_title(f"{title} (top {displayed_nodes} nós)", fontsize=font_size+4, fontweight='bold')
            ax.axis('off')
            
            # Ajusta margens para evitar cortes
            plt.tight_layout(pad=2.0)
            
            # Exibe no Streamlit
            st.pyplot(fig)
            plt.close(fig)
            
            # Legenda com parâmetros
            st.caption(f"📌 Estático | Nós: {node_metric} | Escala: {node_scale:.1f} | Arestas: peso×{edge_scale:.1f} | Fonte: {font_size}pt")
            
        except Exception as e:
            st.error(f"❌ Erro ao gerar visualização estática: {e}")