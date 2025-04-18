import baseline.shortestDistances as sd

def updatedShortestDistanceOverAnEdge(edgeNodeDistances):
    res = {}
    for edge in edgeNodeDistances:
        for node in edgeNodeDistances[edge]:
            if node not in res:
                res[node] = edgeNodeDistances[edge][node]
            elif edgeNodeDistances[edge][node] < res[node]:
                res[node] = edgeNodeDistances[edge][node]
    return res

def updatedNodePositionToXijPosition(updatedShortestDistanceOverAnEdge, shortestDistances, nodes):
    res = {}
    current = 0
    for node in updatedShortestDistanceOverAnEdge:
        res[nodes.index(node)] = (current, shortestDistances[node] - updatedShortestDistanceOverAnEdge[node], updatedShortestDistanceOverAnEdge[node])
        current = current + shortestDistances[node] - updatedShortestDistanceOverAnEdge[node]
    return res, current
def getNoteToGenderNX(G):
    nodeToGender = {}
    for node in list(G.nodes()):
        if G.nodes()[node]['gender'] == 'male':
            nodeToGender[node] = "1"
        else:
            nodeToGender[node] = "2"
    return nodeToGender

def getGroupsNames(nodeToGender, nodes):
    g1 = []
    g2 = []
    for node in nodes:
        if nodeToGender[node] == '1':
            g1.append(node)
        else:
            g2.append(node)
    return g1, g2

def getGroups(nodeToGender, nodes):
    g1 = []
    g2 = []
    for node in range(len(nodes)):
        if nodeToGender[nodes[node]] == '1':
            g1.append(node)
        else:
            g2.append(node)
    return g1, g2

def getNodeNeighbors(nodes, edges):
    nodeNeighbors = {}
    for node in range(len(nodes)):
        nodeNeighbors[node] = []
        for edge in range(len(edges)):
            if edges[edge][0] == nodes[node]:
                nodeNeighbors[node].append(edges[edge][1])
            if edges[edge][1] == nodes[node]:
                nodeNeighbors[node].append(edges[edge][0])
    return nodeNeighbors

def edgeToNodeDistancesUpdated(G, edges, shortestDistances):
    res = {}
    for edge in range(len(edges)):
        if shortestDistances[edges[edge][0]] + 1 < shortestDistances[edges[edge][1]]:
            #print(edge,edges[edge])
            G.add_edge(edges[edge][0], edges[edge][1])
            res[edge] = sd.shortestDistancesNewEdges(G, [edges[edge]], shortestDistances)
            G.remove_edge(edges[edge][0], edges[edge][1])
        else:
            res[edge] = {}
    return res