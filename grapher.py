import os
import cv2
import pandas as pd
import numpy as np
import networkx as nx
from utils import get_text_features

def read_label(path):
    df = []
    with open(path, 'r') as f:
        labels = f.read().splitlines()
    for label in labels:
        xmin, ymin, _, _, xmax, ymax, _, _ = label.split(',')[:8]
        text = ','.join(label.split(',')[8:])
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        df.append([xmin, ymin, xmax, ymax, text])
    return pd.DataFrame(df, columns=['xmin', 'ymin', 'xmax', 'ymax', 'Object'])

class ObjectTree():
    def __init__(self, label_column='label'):
        self.label_column = label_column 
        self.df = None
        self.img = None 
        self.count = 0
    
    def read(self, object_map, image):
        assert image.ndim == 2, 'Check image is grayscale'

        required_cols = {'xmin', 'ymin', 'xmax', 'ymax', 'Object', self.label_column}
        un_required_cols = set(object_map.columns) - required_cols
        object_map.drop(un_required_cols, inplace=True)
        self.df = object_map 
        self.img = image 
        # print(self.df.head())
    
    def connect(self, plot=False, export_df=False):
        df, img = self.df, self.img 
        try:
            if len(df) == 0:
                return 
        except:
            return 

        df_plot = pd.DataFrame()

        distances, nearest_dest_ids_vert = [], []
        x_src_coords_vert, y_src_coords_vert, x_dest_coords_vert, y_dest_coords_vert = [], [], [], []
        lengths, nearest_dest_ids_hori = [], []
        x_src_coords_hori, y_src_coords_hori, x_dest_coords_hori, y_dest_coords_hori = [], [], [], []

        for src_idx, src_row in df.iterrows():
            src_range_x = (src_row['xmin'], src_row['xmax'])
            src_center_y = (src_row['ymin'] + src_row['ymax']) // 2 
            dest_attr_vert = []

            src_range_y = (src_row['ymin'], src_row['ymax'])
            src_center_x  =(src_row['xmin'] + src_row['xmax']) // 2
            dest_attr_hori = []

            for dest_idx, dest_row in df.iterrows():
                is_beneath = False 
                if not src_idx == dest_idx:
                    dest_range_x = (dest_row['xmin'], dest_row['xmax'])
                    dest_center_y = (dest_row['ymin'] + dest_row['ymax']) / 2

                    height = dest_center_y - src_center_y
                    if dest_center_y > src_center_y:
                        if dest_range_x[0] <= src_range_x[0] and dest_range_x[1] >= src_range_x[1]:
                            x_common = (src_range_x[0] + src_range_x[1]) / 2
                            line_src = (x_common, src_center_y)
                            line_dest = (x_common, dest_center_y)
                            attributes = (dest_idx, line_src, line_dest, height)
                            dest_attr_vert.append(attributes)
                            is_beneath = True 
                        elif dest_range_x[0] >= src_range_x[0] and dest_range_x[1] <= src_range_x[1]:
                            x_common = (dest_range_x[0] + dest_range_x[1]) / 2
                            line_src = (x_common, src_center_y)
                            line_dest = (x_common, dest_center_y)
                            attributes = (dest_idx, line_src, line_dest, height)
                            dest_attr_vert.append(attributes)
                            is_beneath = True
                        elif dest_range_x[0] <= src_range_x[0] and dest_range_x[1] >= src_range_x[0] and dest_range_x[1] < src_range_x[1]:
                            x_common = (src_range_x[0] + dest_range_x[1]) / 2
                            line_src = (x_common, src_center_y)
                            line_dest = (x_common, dest_center_y)
                            attributes = (dest_idx, line_src, line_dest, height)
                            dest_attr_vert.append(attributes)
                            is_beneath = True 
                        elif dest_range_x[0] <= src_range_x[1] and dest_range_x[1] >= src_range_x[1] and dest_range_x[0] > src_range_x[0]:
                            x_common = (dest_range_x[0] + src_range_x[1]) / 2
                            line_src = (x_common, src_center_y)
                            line_dest = (x_common, dest_center_y)
                            attributes = (dest_idx, line_src, line_dest, height)
                            dest_attr_vert.append(attributes)
                            is_beneath = True 
                    
                if not is_beneath:
                    dest_range_y = (dest_row['ymin'], dest_row['ymax'])
                    dest_center_x = (dest_row['xmin'] + dest_row['xmax']) / 2
                    if dest_center_x > src_center_x:
                        length = dest_center_x - src_center_x
                    else:
                        length = 0

                    if dest_center_x > src_center_x:
                        if dest_range_y[0] >= src_range_y[0] and dest_range_y[1] <= src_range_y[1]:
                            y_common = (dest_range_y[0] + dest_range_y[1]) / 2
                            line_src = (src_center_x, y_common)
                            line_dest = (dest_center_x, y_common)
                            attributes = (dest_idx, line_src, line_dest, length)
                            dest_attr_hori.append(attributes)
                        if dest_range_y[0] <= src_range_y[0] and dest_range_y[1] <= src_range_y[1] and dest_range_y[1] > src_range_y[0]:
                            y_common = (src_range_y[0] + dest_range_y[1]) / 2
                            line_src = (src_center_x, y_common)
                            line_dest = (dest_center_x, y_common)
                            attributes = (dest_idx, line_src, line_dest, length)
                            dest_attr_hori.append(attributes)
                        if dest_range_y[0] >= src_range_y[0] and dest_range_y[1] >= src_range_y[1] and dest_range_y[0] < src_range_y[1]:
                            y_common = (dest_range_y[0] + src_range_y[1]) / 2
                            line_src = (src_center_x, y_common)
                            line_dest = (dest_center_x, y_common)
                            attributes = (dest_idx, line_src, line_dest, length)
                            dest_attr_hori.append(attributes)
                        if dest_range_y[0] <= src_range_y[0] and dest_range_y[1] >= src_range_y[1]:
                            y_common = (src_range_y[0] + src_range_y[1]) / 2
                            line_src = (src_center_x, y_common)
                            line_dest = (dest_center_x, y_common)
                            attributes = (dest_idx, line_src, line_dest, length)
                            dest_attr_hori.append(attributes)
                        
            dest_attr_vert_sorted = sorted(dest_attr_vert, key=lambda x: x[3])
            dest_attr_hori_sorted = sorted(dest_attr_hori, key=lambda x: x[3])

            if len(dest_attr_vert_sorted) == 0:
                # pass
                nearest_dest_ids_vert.append(-1)
                x_src_coords_vert.append(-1)
                y_src_coords_vert.append(-1)
                x_dest_coords_vert.append(-1)
                y_dest_coords_vert.append(-1)
                distances.append(0)
            else:
                nearest_dest_ids_vert.append(dest_attr_vert_sorted[0][0])
                x_src_coords_vert.append(dest_attr_vert_sorted[0][1][0])
                y_src_coords_vert.append(dest_attr_vert_sorted[0][1][1])
                x_dest_coords_vert.append(dest_attr_vert_sorted[0][2][0])
                y_dest_coords_vert.append(dest_attr_vert_sorted[0][2][1])
                distances.append(dest_attr_vert_sorted[0][3])

            if len(dest_attr_hori_sorted) == 0:
                # pass
                nearest_dest_ids_hori.append(-1)
                x_src_coords_hori.append(-1)
                y_src_coords_hori.append(-1)
                x_dest_coords_hori.append(-1)
                y_dest_coords_hori.append(-1)
                lengths.append(0)
            else:
                try:
                    nearest_dest_ids_hori.append(dest_attr_hori_sorted[0][0])
                except:
                    nearest_dest_ids_hori.append(-1)
                try:
                    x_src_coords_hori.append(dest_attr_hori_sorted[0][1][0])
                except:
                    x_src_coords_hori.append(-1)
                try:
                    y_src_coords_hori.append(dest_attr_hori_sorted[0][1][1])
                except:
                    y_src_coords_hori.append(-1)
                try:
                    x_dest_coords_hori.append(dest_attr_hori_sorted[0][2][0])
                except:
                    x_dest_coords_hori.append(-1)
                try:
                    y_dest_coords_hori.append(dest_attr_hori_sorted[0][2][1])
                except:
                    y_dest_coords_hori.append(-1)
                try:
                    lengths.append(dest_attr_hori_sorted[0][3])
                except:
                    lengths.append(0)

        # print(len(nearest_dest_ids_vert), len(df), nearest_dest_ids_vert)
        # print(df.head())
        # df['below_object'] = df.loc[nearest_dest_ids_vert, 'Object'].values
        df['below_object'] = df.reindex(nearest_dest_ids_vert)['Object'].values
        df['below_dist'] = distances 
        df['below_obj_index'] = nearest_dest_ids_vert
        df_plot['x_src_vert'] = x_src_coords_vert
        df_plot['y_src_vert'] = y_src_coords_vert
        df_plot['x_dest_vert'] = x_dest_coords_vert
        df_plot['y_dest_vert'] = y_dest_coords_vert

        # df['side_object'] = df.loc[nearest_dest_ids_hori, 'Object'].values 
        df['side_object'] = df.reindex(nearest_dest_ids_hori)['Object'].values 
        df['side_length'] = lengths
        df['side_obj_index'] = nearest_dest_ids_hori
        df_plot['x_src_hori'] =  x_src_coords_hori 
        df_plot['y_src_hori'] = y_src_coords_hori
        df_plot['x_dest_hori'] = x_dest_coords_hori 
        df_plot['y_dest_hori'] = y_dest_coords_hori 

        df_merged = pd.concat([df, df_plot], axis=1)

        groups_vert = df_merged.groupby('below_obj_index')['below_dist'].min()
        groups_dict_vert = dict(zip(groups_vert.index, groups_vert.values))
        groups_hori = df_merged.groupby('side_obj_index')['side_length'].min()
        groups_dict_hori = dict(zip(groups_hori.index, groups_hori.values))

        revised_distances_vert = []
        revised_distances_hori = []
        rev_x_src_vert, rev_y_src_vert, rev_x_dest_vert, rev_y_dest_vert = [], [], [], []
        rev_x_src_hori, rev_y_src_hori, rev_x_dest_hori, rev_y_dest_hori = [], [], [], [] 

        for idx, row in df_merged.iterrows():
            below_idx = row['below_obj_index']
            side_idx = row['side_obj_index']
            if row['below_dist'] > groups_dict_vert[below_idx]:
                revised_distances_vert.append(-1)
                rev_x_src_vert.append(-1)
                rev_y_src_vert.append(-1)
                rev_x_dest_vert.append(-1)
                rev_y_dest_vert.append(-1)
            else:
                revised_distances_vert.append(row['below_dist'])
                rev_x_src_vert.append(row['x_src_vert'])
                rev_y_src_vert.append(row['y_src_vert'])
                rev_x_dest_vert.append(row['x_dest_vert'])
                rev_y_dest_vert.append(row['y_dest_vert'])
            
            if row['side_length'] > groups_dict_hori[side_idx]:
                revised_distances_hori.append(-1)
                rev_x_src_hori.append(-1)
                rev_y_src_hori.append(-1)
                rev_x_dest_hori.append(-1)
                rev_y_dest_hori.append(-1)
            else:
                revised_distances_hori.append(row['side_length'])
                rev_x_src_hori.append(row['x_src_hori'])
                rev_y_src_hori.append(row['y_src_hori'])
                rev_x_dest_hori.append(row['x_dest_hori'])
                rev_y_dest_hori.append(row['y_dest_hori'])
            
        df['revised_distances_vert'] = revised_distances_vert
        df_merged['x_src_vert'] = rev_x_src_vert
        df_merged['y_src_vert'] = rev_y_src_vert 
        df_merged['x_dest_vert'] = rev_x_dest_vert 
        df_merged['y_dest_vert'] = rev_y_dest_vert

        df['revised_distances_hori'] = revised_distances_hori
        df_merged['x_src_hori'] = rev_x_src_hori
        df_merged['y_src_hori'] = rev_y_src_hori 
        df_merged['x_dest_hori'] = rev_x_dest_hori 
        df_merged['y_dest_hori'] = rev_y_dest_hori
    
        if plot:
            os.makedirs('grapher_output', exist_ok=True) 
            # try:
            #     if len(img) == None:
            #         pass 
            # except:
            #     pass 

            if img is not None:
                for idx, row in df_merged.iterrows():
                    cv2.line(img, (int(row['x_src_vert']), int(row['y_src_vert'])), (int(row['x_dest_vert']), int(row['y_dest_vert'])), (0, 0, 255), 2)
                    cv2.line(img, (int(row['x_src_hori']), int(row['y_src_hori'])), (int(row['x_dest_hori']), int(row['y_dest_hori'])), (0, 0, 255), 2)
                cv2.imwrite('grapher_output/img.jpg', img)


        if export_df:
            pass 

        graph_dict = {}
        for src_id, row in df.iterrows():
            if row['below_obj_index'] != -1:
                graph_dict[src_id] = [row['below_obj_index']]
            if row['side_obj_index'] != -1:
                graph_dict[src_id] = [row['side_obj_index']]
        
        return graph_dict, df['Object'].tolist()


class Graph():
    def __init__(self, max_nodes=50):
        self.max_nodes = max_nodes 
    
    def _pad_adj(self, adj):
        assert adj.shape[0] == adj.shape[1], "Not square"
        n = adj.shape[0]
        if n < self.max_nodes:
            target = np.zeros(shape=(self.max_nodes, self.max_nodes))
            target[:n, :n] = adj 
        elif n > self.max_nodes:
            target = adj[:self.max_nodes, :self.max_nodes]
        else:
            target = adj 
        return target 
    
    def _pad_text_features(self, fear_arr):
        target = np.zeros(shape=(self.max_nodes, fear_arr.shape[1]))
        if self.max_nodes > fear_arr.shape[0]:
            target[:fear_arr.shape[0], :fear_arr.shape[1]] = fear_arr 
        elif self.max_nodes < fear_arr.shape[0]:
            target = fear_arr[:self.max_nodes, fear_arr.shape[1]]
        else:
            target = fear_arr 
        return target 

    def make_graph_data(self, graph_dict, text_list):
        G = nx.from_dict_of_lists(graph_dict)
        adj_sparse = nx.adjacency_metrix(G)

        A = np.array(adj_sparse.todense())
        A = self._pad_adj(A)

        fear_list = list(map(get_text_features, text_list))
        fear_arr = np.array(fear_list)
        X = self._pad_text_features(fear_arr)
        return A, X



data_dir = '../data/SROIE/task1_train/0325updated.task1train(626p)'

paths = os.listdir(data_dir)
print(len(paths))

label_paths = [path for path in paths if path.endswith('txt')]
tree = ObjectTree()
for path in label_paths:
    df = read_label(os.path.join(data_dir, path))
    im = cv2.imread(os.path.join(data_dir, path.replace('txt', 'jpg')), 0)
    print(df.head())

    tree.read(df, im)
    graph_dict, text_list = tree.connect(plot=True, export_df=False)
    print(text_list)
    break