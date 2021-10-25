class Vertex(object):
    def __init__(self, key):
        self.id = key 							#初始化顶点的键
        self.connectedTo = {}					#初始化顶点的值

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])


    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self, nbr):
        return self.connectedTo[nbr]



class Graph(object):
    def __init__(self):
        self.vertlist = {}
        self.numVertices = 0

    def addVertex(self , key):
        newVertex = Vertex(key)
        self.vertlist[key] = newVertex
        self.numVertices = self.numVertices + 1

    def getVertex(self , n):
        if n in self.vertlist:
            return self.vertlist[n]
        else:
            return None

    def __contains__(self, n):
        return n in self.vertlist

    def addEdge(self , f , t , cost = 0):
        if f not in self.vertlist:
            self.addVertex(f)
        if t not in self.vertlist:
            self.addVertex(t)
        self.vertlist[f].addNeighbor(self.vertlist[t] , cost)

        # 获取邻接表中所有顶点的键
        def getVertices(self):
            return self.vertList.keys()

        # 迭代显示邻接表的每个顶点的邻居节点
        def __iter__(self):
            return iter(self.vertList.values())

g = Graph() 									#实例化图类
for i in range(6):
	g.addVertex(i) 								#给邻接表添加节点
print(g.vertlist)								#打印邻接表
g.addEdge(0, 1, 5) 								#给邻接表添加边及权重
g.addEdge(0, 5, 2)
g.addEdge(1, 2, 4)
g.addEdge(2, 3, 9)
g.addEdge(3, 4, 7)
g.addEdge(3, 5, 3)
g.addEdge(4, 0, 1)
g.addEdge(5, 4, 8)
g.addEdge(5, 2, 1)
for v in g:
	for w in v.getConnections():
		print("(%s, %s)" % (v.getId(), w.getId()))

