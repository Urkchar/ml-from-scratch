class Node():
    def __init__(self, decision, data):
        self.decision = decision
        self.data = data
        self.children = []

    def add_child(self, node: "Node"):
        self.children.append(node)
