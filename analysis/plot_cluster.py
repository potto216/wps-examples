import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import mplcursors

def main():
    # Get the home directory of the current user
    home_dir = Path.home()

    csv_file = 'nRF528240DK_Google_Nest_Hub_c1_decode_s1_802.15.4_802.15.4_MAC_short.csv'
    full_path_for_csv_file = home_dir / 'data' / 'matter' / 'nordic' / csv_file

    df = pd.read_csv(full_path_for_csv_file)
    # Extract the numeric part from the '+RSSI (802.15.4 Metadata)' column
    df["RSSI"] = df["+RSSI (802.15.4 Metadata)"].str.extract(r'([-+]?\d*\.\d+|\d+)')[0].astype(float)
    
    # Replace blank or NaN Destination PAN ID values with "Not Known"
    df['Dest'] = df['Dest'].fillna("None")
    df['Dest'] = df['Dest'].replace("", "None")
    
    df['Src'] = df['Src'].fillna("None")
    df['Src'] = df['Src'].replace("", "None")
    
    df.to_csv('nRF528240DK_Google_Nest_Hub_c1_decode_s1_802.15.4_802.15.4_MAC_short_debug_01.csv')
    
    
    
    # Create a directed graph (nx.DiGraph); change to nx.Graph() for an undirected view
    G = nx.DiGraph()

    # Assign a unique color for each unique PAN ID
    pan_ids = df['Destination PAN ID'].unique()
    color_map = {}
    cmap = plt.get_cmap('tab10', len(pan_ids))
    for i, pan in enumerate(pan_ids):
        color_map[pan] = cmap(i)

    # Build the node list. We assume devices appear as a source or a destination.
    devices = set(df['Src'].unique()).union(set(df['Dest'].unique()))
    for device in devices:
        # Try to get PAN ID from destination role; then check source. Use "Not Known" if missing.
        pan_series = df[df['Dest'] == device]['Destination PAN ID']
        if not pan_series.empty:
            pan = pan_series.iloc[0]
        else:
            pan_series_src = df[df['Src'] == device]['Destination PAN ID']
            pan = pan_series_src.iloc[0] if not pan_series_src.empty else "Not Known"
        if pd.isna(pan) or pan == "":
            pan = "Not Known"
        G.add_node(device, pan=pan)

    # Add edges for each communication record.
    # If a pair communicates multiple times, average the RSSI.
    for idx, row in df.iterrows():
        src = row['Src']
        dest = row['Dest']
        rssi = row["RSSI"]
        if G.has_edge(src, dest):
            old_rssi = G[src][dest]['rssi']
            new_rssi = (old_rssi + rssi) / 2.0
            G[src][dest]['rssi'] = new_rssi
        else:
            G.add_edge(src, dest, rssi=rssi)

    # Prepare node colors with a default if the node PAN ID is not in color_map.
    default_color = 'gray'
    default_pan = "Not Known"
    nodes_list = list(G.nodes())
    node_colors = [color_map.get(G.nodes[node].get('pan', default_pan), default_color)
                   for node in nodes_list]

    # Map the RSSI values to edge widths.
    edge_widths = []
    for u, v in G.edges():
        rssi = G[u][v]['rssi']
        width = 1 + (abs(-100) - abs(rssi)) / 10.0  # adjust scaling as needed
        edge_widths.append(width)

    # Compute a layout (spring layout positions strongly connected nodes closer)
    pos = nx.spring_layout(G, weight=None)

    plt.figure(figsize=(12, 8))
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)
    nx.draw_networkx_edges(G, pos, width=edge_widths, arrowstyle='-|>', arrowsize=12)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    # Draw edge labels for RSSI (placed roughly in the middle of each edge)
    edge_labels = {(u, v): f"{G[u][v]['rssi']:.1f} dBm" for u, v in G.edges()}
    # remove from edge_labels any keys that have the same value for u and v
    edge_labels = {k: v for k, v in edge_labels.items() if k[0] != k[1]}
    
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5)

    # Add interactive tooltips for nodes with mplcursors.
    # They display the node's name, PAN ID, and a placeholder for additional info.
    cursor = mplcursors.cursor(nodes, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        node = nodes_list[sel.index]
        pan = G.nodes[node].get('pan', "Not Known")
        # Replace "Packets: Info here" with any additional node-specific info if available.
        sel.annotation.set_text(f"Node: {node}\nPAN ID: {pan}\nPackets: Info here")

    plt.title("802.15.4 Device Communication Clusters")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()