# ui/graph_viewer.py
# ============================================================
# VERSÃO COM DESTAQUE DAS ARESTAS MAIS FORTES
# ============================================================

import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from src.graph_builder import filter_top_nodes_by_criterion

def prepare_node_attributes(G, metric='degree', scale_factor=1.0, min_size=10, max_size=80):
    """Adiciona atributos 'size' e 'color' aos nós."""
    if G is None or G.number_of_nodes() == 0:
        return G
    
    G_copy = G.copy()
    
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
        values = {}
        for node in G_copy.nodes:
            freq = G_copy.nodes[node].get('frequency', 0)
            values[node] = freq if freq > 0 else G_copy.degree(node)
    else:
        values = {node: 1 for node in G_copy.nodes}
    
    max_val = max(values.values()) if values else 1
    min_val = min(values.values()) if values else 0
    range_val = max_val - min_val if max_val != min_val else 1
    
    for node in G_copy.nodes:
        raw = values.get(node, 0)
        norm = (raw - min_val) / range_val
        size = min_size + norm * (max_size - min_size)
        size = size * scale_factor
        G_copy.nodes[node]['size'] = max(min_size * 0.5, min(max_size * 1.5, size))
    
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
    top_n=30,
    filter_criterion='degree',
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
    label_trim=25,
    top_edges_to_highlight=10  # <-- NOVO PARÂMETRO
):
    """
    Exibe grafo estático (Matplotlib) com destaque das arestas mais fortes.
    """
    if G is None or G.number_of_nodes() == 0:
        st.info(f"ℹ️ Sem dados para gerar o grafo '{title}'.")
        return

    G_filtered = filter_top_nodes_by_criterion(G, top_n, filter_criterion)
    if G_filtered is None or G_filtered.number_of_nodes() < 2:
        st.info(f"ℹ️ Grafo muito pequeno para visualização (menos de 2 nós).")
        return

    if G_filtered.nodes and 'size' not in G_filtered.nodes[next(iter(G_filtered.nodes))]:
        G_filtered = prepare_node_attributes(
            G_filtered,
            metric=node_metric,
            scale_factor=node_scale,
            min_size=node_min_size,
            max_size=node_max_size
        )
    else:
        for node in G_filtered.nodes:
            if 'size' in G_filtered.nodes[node]:
                G_filtered.nodes[node]['size'] *= node_scale

    total_nodes = G.number_of_nodes()
    displayed_nodes = G_filtered.number_of_nodes()
    st.caption(f"📊 Exibindo {displayed_nodes} nós (dos {total_nodes} totais).")

    try:
        # --- Coleta tamanhos e cores dos nós ---
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

        # --- Separa arestas por peso para destaque ---
        edges = list(G_filtered.edges(data=True))
        if edges:
            # Ordena arestas por peso decrescente
            edges_sorted = sorted(edges, key=lambda x: x[2].get('weight', 0), reverse=True)
            
            # Separa top N e o restante
            highlight_count = min(top_edges_to_highlight, len(edges_sorted))
            edges_highlight = edges_sorted[:highlight_count]
            edges_other = edges_sorted[highlight_count:]
            
            # Define espessura e cor para arestas destacadas
            # Pega os pesos das arestas destacadas para ajustar a espessura
            if edges_highlight:
                weights_highlight = [e[2].get('weight', 1) for e in edges_highlight]
                max_w_highlight = max(weights_highlight) if weights_highlight else 1
                edge_widths_highlight = [
                    edge_min_width + (w / max_w_highlight) * (edge_max_width - edge_min_width) * 1.2
                    for w in weights_highlight
                ]
                edge_widths_highlight = [w * edge_scale for w in edge_widths_highlight]
                edge_widths_highlight = [max(0.5, min(8.0, w)) for w in edge_widths_highlight]
                edge_colors_highlight = ['#FF6B00'] * len(edges_highlight)  # laranja vibrante
            else:
                edge_widths_highlight = []
                edge_colors_highlight = []
            
            # Arestas não destacadas: cor cinza claro e transparentes
            if edges_other:
                weights_other = [e[2].get('weight', 1) for e in edges_other]
                max_w_other = max(weights_other) if weights_other else 1
                edge_widths_other = [
                    edge_min_width + (w / max_w_other) * (edge_max_width - edge_min_width) * 0.6
                    for w in weights_other
                ]
                edge_widths_other = [w * edge_scale for w in edge_widths_other]
                edge_widths_other = [max(0.2, min(4.0, w)) for w in edge_widths_other]
                edge_colors_other = ['#CCCCCC'] * len(edges_other)
            else:
                edge_widths_other = []
                edge_colors_other = []
            
            # --- Construir listas para desenho ---
            edge_lists = []
            edge_widths = []
            edge_colors = []
            
            # Adiciona arestas destacadas primeiro (para ficarem por cima)
            for i, (u, v, data) in enumerate(edges_highlight):
                edge_lists.append((u, v))
                edge_widths.append(edge_widths_highlight[i])
                edge_colors.append(edge_colors_highlight[i])
            
            # Adiciona arestas não destacadas
            for i, (u, v, data) in enumerate(edges_other):
                edge_lists.append((u, v))
                edge_widths.append(edge_widths_other[i])
                edge_colors.append(edge_colors_other[i])
        else:
            edge_lists = []
            edge_widths = []
            edge_colors = []

        # --- Layout ---
        if layout_type == 'spring':
            pos = nx.spring_layout(G_filtered, seed=42, k=layout_k, iterations=layout_iterations)
        elif layout_type == 'circular':
            pos = nx.circular_layout(G_filtered)
        elif layout_type == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G_filtered)
        elif layout_type == 'shell':
            pos = nx.shell_layout(G_filtered)
        else:
            pos = nx.random_layout(G_filtered, seed=42)

        # --- Cria figura ---
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=120)
        ax.set_facecolor('#FAFAFA')
        fig.patch.set_facecolor('#FAFAFA')

        # --- Desenha arestas (primeiro as não destacadas, depois as destacadas) ---
        # Desenha arestas não destacadas
        if edges_other:
            nx.draw_networkx_edges(
                G_filtered, pos, ax=ax,
                edgelist=[(u, v) for u, v in edge_lists[len(edges_highlight):]],
                alpha=0.3, width=edge_widths_other, edge_color='#CCCCCC'
            )
        
        # Desenha arestas destacadas (por cima, com cor vibrante)
        if edges_highlight:
            nx.draw_networkx_edges(
                G_filtered, pos, ax=ax,
                edgelist=[(u, v) for u, v in edge_lists[:len(edges_highlight)]],
                alpha=0.9, width=edge_widths_highlight, edge_color='#FF6B00'
            )

        # --- Desenha nós ---
        nx.draw_networkx_nodes(
            G_filtered, pos, ax=ax,
            node_size=sizes, node_color=colors, alpha=0.85,
            edgecolors='#333333', linewidths=0.8
        )

        # --- Rótulos com halo ---
        if show_labels:
            for node, (x, y) in pos.items():
                label = labels.get(node, '')
                if label:
                    ax.text(x, y, label, fontsize=font_size, fontweight='bold',
                            color='white', ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.0))
            nx.draw_networkx_labels(
                G_filtered, pos, ax=ax,
                labels=labels, font_size=font_size, font_weight='bold',
                font_color='#222222'
            )

        # --- Título e legenda ---
        ax.set_title(
            f"{title} (top {displayed_nodes} nós • {G_filtered.number_of_edges()} arestas)",
            fontsize=font_size + 6,
            fontweight='bold',
            pad=20
        )
        ax.axis('off')
        plt.tight_layout(pad=2.0)

        # --- Exibe ---
        st.pyplot(fig)
        plt.close(fig)

        # --- Legenda com informações sobre destaque ---
        if top_edges_to_highlight > 0 and edges_highlight:
            st.caption(
                f"📌 Estático | Nós: {node_metric} | Escala: {node_scale:.1f}x | "
                f"Arestas destacadas: {len(edges_highlight)} (laranja) | Demais: cinza"
            )
        else:
            st.caption(
                f"📌 Estático | Nós: {node_metric} | Escala: {node_scale:.1f}x | "
                f"Arestas: {edge_scale:.1f}x | Fonte: {font_size}pt | Layout: {layout_type}"
            )

    except Exception as e:
        st.error(f"❌ Erro ao gerar visualização: {e}")