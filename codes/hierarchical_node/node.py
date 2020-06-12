class Node:
    def __init__(self, id, direct_children, all_children, path, level):
        self.id = id
        self.direct_children = direct_children
        self.all_children = all_children
        self.path = path
        self.level = level

    def __str__(self):
        str = "id: %s\n" % self.id
        str+= "direct_children %s \n" % self.direct_children
        str+= "all_children %s \n" % self.all_children
        str+= "path %s \n" % self.path
        str+= "level %s \n" % self.level