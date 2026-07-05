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

def prepare_node_attributes(G, metric='degree', scale_factor=1.0, min_size=10, max_size=80, log_scale=True):
    """
    Adiciona atributos 'size' e 'color' aos nós.
    - size: calculado automaticamente a partir da métrica, com mapeamento entre min_size e max_size.
    - scale_factor: multiplicador global para ajuste fino.
    - log_scale: se True, usa escala logarítmica para distribuir os tamanhos.
    """
    if G is None or G.number_of_nodes() == 0:
        return G
    
    G_copy = G.copy()
    
    # Coleta os valores da métrica escolhida
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
    elif metric == 'frequency':
        # Usa o atributo 'frequency' se existir, senão grau
        values = {}
        for node in G_copy.nodes:
            freq = G_copy.nodes[node].get('frequency', 0)
            values[node] = freq if freq > 0 else G_copy.degree(node)
    else:  # 'uniform'
        values = {node: 1 for node in G_copy.nodes}
    
    # Aplica escala logarítmica se solicitado e houver valores positivos
    if log_scale:
        min_val = min(values.values()) if values else 0
        # Evita log de zero
        values_log = {}
        for node, val in values.items():
            if val <= 0:
                values_log[node] = 0
            else:
                values_log[node] = np.log1p(val)  # log(1+x) para valores pequenos
        values = values_log
    
    # Normaliza e mapeia para o intervalo [min_size, max_size]
    max_val = max(values.values()) if values else 1
    min_val = min(values.values()) if values else 0
    range_val = max_val - min_val if max_val != min_val else 1
    
    for node in G_copy.nodes:
        raw = values.get(node, 0)
        norm = (raw - min_val) / range_val
        size = min_size + norm * (max_size - min_size)
        size = size * scale_factor
        # Garante limites
        size = max(min_size * 0.5, min(max_size * 1.5, size))
        G_copy.nodes[node]['size'] = size
    
    # Cor (comunidades ou uniforme) - mantido
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

import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyvis.network import Network
import streamlit.components.v1 as components
import os

def display_graph(
    G,
    title,
    key_suffix,
    view_mode,
    top_n=30,
    filter_criterion='degree',
    node_metric='degree',
    node_scale=1.0,
    node_min_size=None,   # se None, calcula automático
    node_max_size=None,   # se None, calcula automático
    edge_scale=1.0,
    edge_min_width=None,  # se None, calcula automático
    edge_max_width=None,  # se None, calcula automático
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
    Exibe grafo com autoajuste de tamanho de nós e espessura de arestas.
    """
    if G is None or G.number_of_nodes() == 0:
        st.info(f"ℹ️ Sem dados para gerar o grafo '{title}'.")
        return

    # Filtra nós conforme critério
    G_filtered = filter_top_nodes_by_criterion(G, top_n, filter_criterion)
    if G_filtered is None or G_filtered.number_of_nodes() < 2:
        st.info(f"ℹ️ Grafo muito pequeno para visualização (menos de 2 nós).")
        return

    # Aplica atributos visuais (size e color) – já deve existir prepare_node_attributes
    # Se o grafo não tiver atributos, aplica padrão
    if G_filtered.nodes and 'size' not in G_filtered.nodes[next(iter(G_filtered.nodes))]:
        G_filtered = prepare_node_attributes(
            G_filtered,
            metric=node_metric,
            scale_factor=node_scale,
            min_size=10 if node_min_size is None else node_min_size,
            max_size=80 if node_max_size is None else node_max_size
        )
    else:
        # Se já tem atributos, aplica apenas o scale_factor
        for node in G_filtered.nodes:
            if 'size' in G_filtered.nodes[node]:
                G_filtered.nodes[node]['size'] *= node_scale

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
                net = Network(height="600px", width="100%", notebook=False, bgcolor="#ffffff")
                net.from_nx(G_filtered)

                # Aplica tamanho e cor
                for node in G_filtered.nodes:
                    if node in net.nodes:
                        net.nodes[node]['size'] = G_filtered.nodes[node].get('size', 30)
                        net.nodes[node]['color'] = G_filtered.nodes[node].get('color', '#4A90D9')

                # Ajusta espessura das arestas (auto + escala)
                edges = G_filtered.edges(data=True)
                weights = [data.get('weight', 1) for _, _, data in edges]
                if weights:
                    max_w = max(weights) if weights else 1
                    min_w = min(weights) if weights else 1
                    # Escala logarítmica se houver grande variação
                    if max_w / (min_w + 1) > 10:
                        log_weights = [np.log1p(w) for w in weights]
                        max_log = max(log_weights) if log_weights else 1
                        edge_widths = [0.5 + (log_w / max_log) * 4.5 for log_w in log_weights]
                    else:
                        edge_widths = [0.5 + (w / max_w) * 4.5 for w in weights]
                    # Aplica multiplicador
                    edge_widths = [w * edge_scale for w in edge_widths]
                    edge_widths = [max(0.2, min(8.0, w)) for w in edge_widths]
                    # Aplica no PyVis
                    for i, (u, v, data) in enumerate(edges):
                        if net.get_edge(u, v):
                            net.edges[net.get_edge(u, v)]['width'] = edge_widths[i]

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
                st.components.v1.html(st.session_state[cache_key], height=650)
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
                         filter_criterion, node_metric, node_scale,
                         node_min_size, node_max_size, edge_scale,
                         edge_min_width, edge_max_width, font_size,
                         fig_width, fig_height, layout_type,
                         layout_k, layout_iterations, show_labels, label_trim)

    # ============================================================
    # MODO ESTÁTICO (Matplotlib) – COM AUTOAJUSTE
    # ============================================================
    else:
        try:
            # --- Coleta tamanhos e cores dos nós ---
            sizes = []
            colors = []
            labels = {}
            for node in G_filtered.nodes:
                size = G_filtered.nodes[node].get('size', 30)
                sizes.append(size)
                colors.append(G_filtered.nodes[node].get('color', '#4A90D9'))
                label = str(node)
                if len(label) > label_trim:
                    label = label[:label_trim-3] + "..."
                labels[node] = label

            # --- Autoajuste de espessura das arestas ---
            edges = G_filtered.edges(data=True)
            weights = [data.get('weight', 1) for _, _, data in edges]
            if weights:
                max_w = max(weights) if weights else 1
                min_w = min(weights) if weights else 1
                # Escala logarítmica se a variação for grande
                if max_w / (min_w + 1) > 10:
                    log_weights = [np.log1p(w) for w in weights]
                    max_log = max(log_weights) if log_weights else 1
                    edge_widths = [0.5 + (log_w / max_log) * 4.5 for log_w in log_weights]
                else:
                    edge_widths = [0.5 + (w / max_w) * 4.5 for w in weights]
                # Aplica multiplicador
                edge_widths = [w * edge_scale for w in edge_widths]
                edge_widths = [max(0.2, min(8.0, w)) for w in edge_widths]
            else:
                edge_widths = [0.5] * len(edges)

            # --- Layout ---
            if layout_type == 'spring':
                pos = nx.spring_layout(G_filtered, seed=42, k=layout_k, iterations=layout_iterations)
            elif layout_type == 'circular':
                pos = nx.circular_layout(G_filtered)
            elif layout_type == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G_filtered)
            else:
                pos = nx.random_layout(G_filtered, seed=42)

            # --- Desenho ---
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            nx.draw_networkx_edges(G_filtered, pos, ax=ax, alpha=0.4, width=edge_widths, edge_color='gray')
            nx.draw_networkx_nodes(G_filtered, pos, ax=ax, node_size=sizes, node_color=colors,
                                   alpha=0.85, edgecolors='black', linewidths=0.5)
            if show_labels:
                nx.draw_networkx_labels(G_filtered, pos, ax=ax, labels=labels, font_size=font_size,
                                        font_weight='bold', font_color='black')

            ax.set_title(f"{title} (top {displayed_nodes} nós)", fontsize=font_size+4, fontweight='bold')
            ax.axis('off')
            plt.tight_layout(pad=2.0)
            st.pyplot(fig)
            plt.close(fig)

            st.caption(f"📌 Estático | Nós: {node_metric} | Escala: {node_scale:.1f} | Arestas: ×{edge_scale:.1f} | Fonte: {font_size}pt")

        except Exception as e:
            st.error(f"❌ Erro ao gerar visualização estática: {e}")