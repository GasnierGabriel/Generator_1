import networkx as nx

social_network = nx.Graph()
social_network.add_node('Fang Bin')
social_network.add_node('Youtube')
social_network.add_node('Twitter')
social_network.add_node('Iqiyi')

social_network.add_edge('Fang Bin', 'Youtube', weight = 8)#
social_network.add_edge('Fang Bin', 'Twitter', weight = 10)
social_network.add_edge('Fang Bin', 'Iqiyi', weight = 1)
social_network.add_edge('Twitter', 'Youtube', weight = 9)


nx.draw(social_network, with_labels=True, font_weight='bold', node_size=5000 , node_color='red')