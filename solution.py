import inspect
import importlib
import re
import random
import hashlib

random.seed(123)

k_bit = 32


class SubClassTypeError(Exception):
    pass


class ArgumentError(Exception):
    pass


class FingerInitError(Exception):
    pass


# Utility functions

def distance(c_hash_a, c_hash_b, k=k_bit):
    """
    Measures the clockwise distance between two consistent hash values.
    :param c_hash_1: int
    :param c_hash_2: int
    :param k: int - the number of bits. defaults to global k_bit
    :return: int - the calculated distance
    """
    dis = c_hash_b % (2 ** k) - c_hash_a % (2 ** k)
    return dis if dis >= 0 else dis + 2 ** k


def question_1():
    # What is the largest possible hash value in the network if k=32?
    print(2 ** 32 - 1)


def c_hash(val, k=k_bit):
    """
    Computes the consistent hash of val using the sha1 hashing algorithm.
    :param val: string - hashable value
    :param k: int - the number of bits. defaults to global k_bit
    :return:int
    """
    # compute the sha1 hash of the val, convert it to an integer, and map it to the circle
    sha1_val = int(hashlib.sha1(val.encode()).hexdigest(), 16)
    return sha1_val % 2 ** k


# Linked List Objects

class Node:
    """
    Node represents the container in which data will be stored in key, value pairs.
    A Node will be part of a circular linked list and will also hold a finger table
    for search optimization over the distributed system.
    """

    def __init__(self, _id, k=k_bit):
        if _id > 2 ** k:
            raise ValueError
        else:
            self._id = _id
            self.__next = None
            self.db = {}
            self.finger_table = []
            self.k = k

    def __repr__(self):
        return f"<Node(id={self._id:,}, k={self.k})>"

    def __str__(self):
        return f"{self._id}"

    @property
    def next(self):
        return self.__next

    def set_next(self, node):
        if not isinstance(node, Node):
            raise TypeError
        self.__next = node

    def print_finger_table(self):
        # This one is a freebie to help you debug
        # Try not to change anything here. Some test cases use this method to get the finger table.
        col_1_size = max(len(str(self.k)), 4)
        col_2_size = max(len(str(2 ** self.k)), 16)
        col_3_size = col_2_size
        total_size = sum([col_1_size, col_2_size, col_3_size]) + 4
        header = f"NODE {str(self)} FINGER TABLE:"
        print(f"{'-' * total_size: ^{total_size}}")
        print(f"{header: ^{total_size}}")
        print(f"{'-' * total_size: ^{total_size}}")
        print(f"{'i' : >{col_1_size}} |"
              f"{'successor hash': >{col_2_size}} |"
              f"{'successor node': >{col_3_size}} ")
        print(f"{'-' * 1: >{col_1_size}} |"
              f"{'-' * 14: >{col_2_size}} |"
              f"{'-' * 14: >{col_3_size}} ")
        for finger in self.finger_table:
            print(f"{finger[0]: >{col_1_size}} |"
                  f"{finger[1]: >{col_2_size}} |"
                  f"{str(finger[2]): >{col_3_size}} ")
        print(f"{'-' * total_size: ^{total_size}}")


class CircularLinkedList:

    def __init__(self, head, k=k_bit):
        self.head = head
        if head:
            self.head.set_next(head)
        self.size = 1
        self.k = head.k if head.k != k_bit else k

    def __repr__(self):
        return f"<{self.__class__.__name__}("f"head={repr(self.head)}, k={self.k}, size={self.size})>"

    def __str__(self):
        """
        A string containing a series of node id's connected by arrows, ->
        e.g. 1 -> 2 -> 3 -> 1
        :return: string
        """
        res = ""
        tmp = self.head
        if self.head:
            while (True):
                res = res + str(tmp) + " -> "
                tmp = tmp.next
                if (tmp == self.head):
                    break
        return res + str(self.head)

    def find_predecessor_node(self, key, hash_key=True):
        """
        Returns the closest node in the counterclockwise direction from the hashed key.
        :param key: string - hashable or already hashed value
        :param hash_key: bool - will hash the key by default unless specified otherwise
        :return: Node
        """
        if hash_key:
            key = c_hash(key,self.k)
        # only one node
        if self.head.next == self.head:
            return self.head

        curr = self.head
        tail = self.head

        while tail.next != self.head:
            tail = tail.next  # get the last node
        # equal to any of node_id case
        while curr.next != self.head:
            if key == curr._id:
                return curr
            curr = curr.next
        if key == curr._id:
            return curr

        curr = self.head
        smallest = tail
        largest = tail
        # if the key is less than smallest or greater than largest, largest is predecessor node
        # usually is the head & tail
        while curr.next != self.head:
            if curr._id < smallest._id:
                smallest = curr
            if curr._id > largest._id:
                largest = curr
            curr = curr.next
        if (key < smallest._id) | (key > largest._id):
            return largest
        # normal interval case
        curr = self.head
        while (curr.next._id < key) & (curr.next != self.head):
            curr = curr.next
        return curr

    def find_successor_node(self, key, hash_key=True):
        """
        Returns the closest node in the clockwise direction from the hashed key.
        :param key: string - hashable or already hashed value
        :param hash_key: bool - will hash the key by default unless specified otherwise
        :return: Node
        """
        if hash_key:
            key = c_hash(key,self.k)
        pre = self.find_predecessor_node(key, False)
        if key == pre._id:
            return pre
        return pre.next

    def insert_node(self, node):
        """
        Adds a new Node sorted by node id to the circular linked list starting from the head.
        If the node id's collide, return the existing node.
        :param node: Node - the new node
        :return: Node - Reference of inserted node.
        """
        self.size += 1
        curr = self.head
        tail = self.head
        if node._id == self.head._id:
            return self.head
        # inserted node has the smallest id, insert it as self.head
        if node._id < self.head._id:
            while tail.next != self.head:
                tail = tail.next
            node.set_next(self.head)
            tail.set_next(node)
            self.head = node
        else:
            while (curr.next._id < node._id) & (curr.next != self.head):
                curr = curr.next
            # collide
            if curr.next._id == node._id:
                return curr.next
            # found position to insert
            node.set_next(curr.next)
            curr.set_next(node)
        return node

    def __iter__(self):
        """
        Creates a Circular Linked List iterator which traverse all the nodes starting and ending at the head node.
        :return: Node
        """
        curr = self.head
        while curr.next != self.head:
            yield curr
            curr = curr.next
        yield curr
        return self.head

    def __next__(self):
        # Your implementation here
        pass

    def traverse_from(self, key=None, steps=None, hash_key=True):
        """
        Find the node responsible for key, and return the node steps away.
        :param key: int or str
        :param steps: int
        :param hash_key: bool
        :return: Node
    Default behavior for key   = None is for the starting node to be the predecessor to key
    Default behavior for steps = None is for the steps to be the size of the list minus 1, such that you traverse the list up to the predecessor node of your starting node.
        """
        if not key:
            node = self.head
        else:
            node = self.find_predecessor_node(key, hash_key)
        if not steps:
            steps = self.size - 1
        for i in range(steps):
            node = node.next
        return node

    def remove_node(self, node_id):
        """
        Finds the node in the system, reallocates its db to the appropriate node, and
        updates each node's finger table.
        :param node_id: int
        :return: bool
        """
        curr = self.head
        prev = self.head
        found_node = None
        prev_node = None
        # need to keep track of previous node of the removed node to update the arrows
        while prev.next != self.head:
            prev = prev.next
        while curr.next != self.head:
            if curr._id == node_id:
                found_node = curr
                prev_node = prev
            curr = curr.next
            prev = prev.next
        # check the tail node
        if (not found_node) & (curr._id == node_id):
            found_node = curr
            prev_node = prev
        # reallocates its db to the next node
        found_node.next.db.update(found_node.db)
        prev_node.set_next(found_node.next)
        self.size -= 1
        return bool(found_node)



class NaiveDistributedHashTable(CircularLinkedList):
    """
    Traverses the distributed system naively. Each node in this system is only aware of its successor node.
    """

    def get(self, key):
        """
        Find and return the data for a given key.
        The containing node is the successor to the key's predecessor.
        :param key: string
        :return: data or None
        """
        key_hash = c_hash(key,self.k)
        ctnr = self.find_successor_node(key_hash,False)
        return ctnr.db[key]


    def put(self, key, data):
        """
        Store the data for the given key.
        The containing node is the successor to the key's predecessor.
        :param key: string
        :param data:
        :return: Node

        If the key_hash equals the node Id, stop - this is the target node.
        Otherwise get the successor node to the key_hash.
        """
        key_hash = c_hash(key,self.k)
        ctnr = self.find_successor_node(key_hash,False)
        ctnr.db[key] = data
        return ctnr


class SmartDistributedHashTable(CircularLinkedList):
    """
    Uses the efficient finger table / chord algorithm to traverse the distributed system.
    """

    def get(self, key):
        """
        Find and return the data for a given key.
        The containing node is the successor to the key's predecessor.
        :param key: string
        :return: data or None
        """

        ctnr = self.find_node(key)
        if key not in ctnr.db.keys():
            return None
        return ctnr.db[key]

    def put(self, key, data):
        """
        Store the data for the given key.
        The containing node is the successor to the key's predecessor.
        :param key: string
        :param data:
        :return: bool
        """
        ctnr = self.find_node(key)
        ctnr.db[key] = data
        return ctnr

    def update_finger_tables(self):
        """
        Updates each node's finger table in the distributed system
        :return: bool
        """

        curr = self.head
        for n in range(self.size):
            curr.finger_table = []
            for i in range(self.k):
                curr.finger_table.append((i,curr._id+2**i, self.find_successor_node(curr._id+2**i, False)))
            curr = curr.next
        return True

    def find_node(self, key, hash_key=True, start_node=None):
        """
        Uses the finger table to get the node responsible for key
        :param key: string - hashable or already hashed value
        :param hash_key: bool - will hash the key by default unless specified otherwise
        :param start_node: Node - assumes the node is already a system member
        :return: Node or None
        """
        # You may want to try recursion here
        if hash_key:
            key = c_hash(key,self.k)

        #initialize
        if not start_node:
            start_node = self.head
        #curr = self.find_successor_node(key,False)
        while True:
            curr = start_node.next
            if (key > start_node._id) and (key <= curr._id):
                return curr
            # search backwards on curr.finger_table
            else:
                for path in start_node.finger_table[::-1]:
                    if path[1] == key:
                        return path[2]
                    if path[1] < key:
                        start_node = path[2]
        return None



def CLLFactory(*, cls=None, k=k_bit, node_count=None, node_ids=[], naive=True):
    """
    Creates and returns a circular linked list of node_count nodes with random ids.
    OR
    Creates and returns a circular linked list of nodes with the provided ids.
    If naive is false, then the factory will also update the node's finger tables.
    :param cls: CircularLinkedList - type of circular linked list to return
    :param k: int - number of bits used to determine size of circular linked list
    :param node_count: int - number of nodes - should not exceed 2**k
    :param node_ids: list[int] - list of all the ids for future nodes - invalid or redundant ids discarded
    :param naive: bool - if false, the factory should update the finger tables
    :return: CircularLinkedList
    """
    # This one is a freebie to help with the test cases... Don't do anything to it unless you know what you are doing!!
    if cls is None:
        cls = CircularLinkedList

    if not issubclass(cls, CircularLinkedList):
        raise SubClassTypeError(f"{cls} is not of type CircularLinkedList")

    if node_count is not None:
        if node_count == 0:
            return None
        if node_count > 2 ** k:
            raise ArgumentError(f"node_count cannot be greater than {2 ** k}")
        used_ids = [random.getrandbits(k)]
        cll = cls(Node(used_ids[0], k=k), k)
        for _ in range(node_count - 1):
            new_id = random.getrandbits(k)
            while new_id in used_ids:
                new_id = random.getrandbits(k)
            cll.insert_node(Node(new_id, k=k))
            used_ids.append(new_id)
        return cll

    if node_ids:
        cll = None
        used_ids = []
        while node_ids:
            _id = node_ids.pop(0)
            if (_id < 2 ** k) and (_id not in used_ids):
                used_ids.append(_id)
                if not cll:
                    cll = cls(Node(_id, k=k), k=k)
                else:
                    cll.insert_node(Node(_id, k=k))
        return cll
    return None


# your turn to test

def test_remove_node():
    # save this test until last. It will be much easier to do then
    '''
    assert False, "Write your own tests for CircularLinkedList.remove_node. Then implement the method. See the docs " \
                  "I wrote for the remove method to get an understanding about what to do with it. Don't cop out " \
                  "here and just assert True. I will be looking at these to determine if you did this. If you " \
                  "don't attempt this problem you will just lose the points allotted to it. If you cop out then you " \
                  "will lose 50% off your final score, whatever it is."
    '''
    n2 = Node(2, k=3)
    dht = SmartDistributedHashTable(n2, k=3)
    n1 = Node(1, k=3)
    n0 = Node(0, k=3)
    n6 = Node(6, k=3)
    for node in [n1, n0, n6]:
        dht.insert_node(node)
    dht.update_finger_tables()
    key = "seb@yahoo.com"
    data = {'class': 'Real-Time Intelligent Systems'}
    found_node = dht.find_node(key)
    found_node.db[key] = data
    found_data = dht.get(key)
    assert found_data == data, "The data is not the same"

    dht.remove_node(6)
    dht.update_finger_tables()
    node0 = dht.find_successor_node(0, hash_key=False)
    node2 = dht.find_successor_node(2, hash_key=False)
    node2_expected_finger_table = [(0, 3, node0), (1, 4, node0), (2, 6, node0)]
    assert node2.finger_table == node2_expected_finger_table, "Your finger table is wrong"


def test_cll_factory():
    sig = inspect.signature(CLLFactory)
    assert sig.__str__() == "(*, cls=None, k=32, node_count=None, node_ids=[], naive=True)", "Signature incorrect"
    positional_argument_error_raised = False
    try:
        CLLFactory(CircularLinkedList)
    except TypeError as e:
        positional_argument_error_raised = "positional arguments" in str(e)
    assert positional_argument_error_raised, "CLLFactory should take no positional arguments"
    cll = CLLFactory(node_count=0)
    assert cll is None
    cll = CLLFactory(node_count=10)
    assert cll.size == 10
    cll = CLLFactory(node_ids=[x for x in range(int(1e6), int(1e8), int(1e7))])
    assert cll.size == 10
    dht = CLLFactory(cls=NaiveDistributedHashTable, node_count=2)
    assert isinstance(dht, NaiveDistributedHashTable)
    dht = CLLFactory(cls=NaiveDistributedHashTable, k=3, node_count=4)
    assert dht.k == 3
    argument_error_raised = False
    try:
        CLLFactory(k=3, node_count=100)
    except ArgumentError:
        argument_error_raised = True
    assert argument_error_raised, "CLLFactory node_count should not exceed 2**k"
    for _ in range(100):
        cll = CLLFactory(k=3, node_count=8)
        assert cll.size == 8, "CLLFactory should guarantee the provided node_count as long as node_count <= 2**k"
    cll = CLLFactory(k=3, node_ids=[10, 9, 8])
    assert cll is None
    cll = CLLFactory(k=3, node_ids=[9, 7])
    assert cll.size == 1
    cll = CLLFactory(k=3, node_ids=[2, 7, 7, 9])
    assert cll.size == 2
    cll = CLLFactory(k=2, node_ids=[0, 1, 2, 3, 3, 3])
    assert cll.size == 4


# test utilities

def test_distance_function():
    error_msg = "Try again - distance(%d, %d)"
    assert distance(0, 100) == 100, error_msg % (0, 100)
    assert distance(100, 1000) == 900, error_msg % (100, 1000)
    assert distance(4e9, 100) == 294967396, error_msg % (4e9, 100)
    assert distance(1, 5, 3) == 4, error_msg % (1, 5)
    assert distance(5, 3, 3) == 6, error_msg % (5, 3)


def test_question_1():
    question_1()


def test_consistent_hashing_algorithm():
    hash1 = c_hash("john@gmail.com")
    hash2 = c_hash("john@gmail.com")
    hash3 = c_hash("jane@yahoo.com")
    assert hash1 == hash2, "hash algorithm is unstable. Hashes of the same value must be equal"
    assert isinstance(hash1, int), "hashing algorithm must return an int"
    assert hash1 <= 2 ** k_bit, "hashing algorithm must implement consistent hashing"
    assert hash1 != hash3, "hashes should be different with high probability"
    hash1 = c_hash("joe@uchicago.edu", k=3)
    assert hash1 <= 2 ** 3, "hashing algorithm must implement consistent hashing to k=3"


# test naive node

def test_node_constructor():
    nid = random.getrandbits(k_bit)
    node = Node(nid)
    assert hasattr(node, '_id'), "Node needs an _id attribute"
    assert hasattr(node, 'next'), "Node needs an next attribute"
    assert hasattr(node, 'db'), "Node needs an db attribute"
    assert hasattr(node, 'finger_table'), "Node needs a finger_table attribute"
    assert hasattr(node, 'k'), "Node needs a k attribute"
    assert node._id == nid, f"Node _id should be {nid}"
    assert node.next is None, f"Node's next should be none because there is no other node at this point"
    assert node.db == {}, "Node needs an empty dictionary to store data"
    assert node.finger_table == [], "Node's fingertable should be an empty list upon creation"
    assert node.k == 32, f"Nodes default k should be {k_bit}"
    node = Node(4, k=3)
    assert node.k == 3
    value_error_raised = False
    try:
        Node(10, k=3)
    except ValueError:
        value_error_raised = True
    assert value_error_raised, "Node _id cannot exceed 2**k"


def test_node_repr():
    node = Node(random.getrandbits(k_bit))
    assert repr(node) == f"<Node(id={node._id:,}, k={k_bit})>"


def test_node_str():
    node = Node(random.getrandbits(k_bit))
    assert str(node) == f"{node._id}", "You didn't define the node __str__ function correctly"


def test_node_set_next():
    n1 = Node(random.getrandbits(k_bit))
    n2 = Node(random.getrandbits(k_bit))
    attribute_error_raised = False
    try:
        n1.next = n2
    except AttributeError:
        attribute_error_raised = True
    assert attribute_error_raised, "Node's next property should not be settable. Better to use set_next."
    assert hasattr(n1, "set_next"), "Node needs a set_next method"

    type_error_raised = False
    try:
        n1.set_next(10)
    except TypeError:
        type_error_raised = True
    assert type_error_raised, "set_next should only accept arguments of type Node and raise a TypeError otherwise"

    n1.set_next(n2)
    assert n1.next == n2, "set_next should update the node's next property"


def test_node_print_finger_table():
    node = Node(4, k=32)
    for i in range(3):
        node.finger_table.append((i, (node._id + 2 ** i) % 2 ** 3, node))
    node.print_finger_table()


# test circular linked list

def test_circular_linked_list_constructor():
    node = Node(random.getrandbits(k_bit))
    cll = CircularLinkedList(node)
    assert node == cll.head, "The passed node should be the head"
    assert cll.head == cll.head.next, "head of circular liked list with one node should point to itself"
    assert cll.size == 1, "The size of newly constructed circular linked list should be 1"
    assert cll.k == k_bit
    cll = CircularLinkedList(Node(1), k=3)
    assert cll.k == 3


def test_circular_linked_list_repr():
    id_val = random.getrandbits(k_bit)
    cll = CircularLinkedList(Node(id_val))
    assert repr(cll) == f"<CircularLinkedList(" \
                        f"head=<Node(id={id_val:,}, k={k_bit})>, k={k_bit}, size={cll.size})>", f"{repr(cll)}"


def test_circular_linked_list_find_predecessor_node():
    head = Node(random.getrandbits(k_bit))
    cll = CircularLinkedList(head)
    rand_key = random.getrandbits(k_bit)
    node = cll.find_predecessor_node(rand_key, hash_key=False)
    assert rand_key != head._id, "The predecessor node to key of a single node circular linked list is the head"
    assert node is head, "The predecessor node to key of a single node circular linked list is the head"
    nodes = []
    current = head
    for x in range(4):
        node = Node(head._id + int(1e8 * (x + 1)))
        nodes.append(node)
        current.set_next(node)
        current = node
    current.set_next(head)
    node = cll.find_predecessor_node(nodes[2]._id, hash_key=False)
    assert node is nodes[2], "A node with id equal to the hashed key is the predecessor node"
    check_key = int((nodes[2]._id + nodes[3]._id) / 2)
    node = cll.find_predecessor_node(check_key, hash_key=False)
    assert node is nodes[2], "The predecessor node of the hashed key should be the closet node in the " \
                             "counterclockwise direction of the hashed key"
    node = cll.find_predecessor_node(1, hash_key=False)
    assert node is nodes[-1], "This case is not unique but will fail if you implement distance and find_predecessor " \
                              "incorrectly"
    node = cll.find_predecessor_node("key")
    assert isinstance(node, Node), "Your find_predecessor function should also be able to hash a key by default"


def test_circular_linked_list_find_successor_node():
    head = Node(100000)
    cll = CircularLinkedList(head)
    nodes = []
    current = head
    for x in range(4):
        node = Node(head._id + int(1e8 * (x + 1)))
        nodes.append(node)
        current.set_next(node)
        current = node
    current.set_next(head)
    check_key = int((nodes[2]._id + nodes[3]._id) / 2)
    node = cll.find_successor_node(check_key, hash_key=False)
    assert node is nodes[3]


def test_circular_linked_list_insert_node():
    head = Node(int(1e3))
    cll = CircularLinkedList(head)
    n1 = cll.insert_node(Node(int(1e7)))
    n2 = cll.insert_node(Node(int(1e4)))
    n3 = cll.insert_node(Node(1))
    assert head.next is n2
    assert n2.next is n1
    assert n1.next is n3
    assert n3.next is head
    assert cll.size == 4
    n4 = cll.insert_node(Node(1))
    assert n4 is n3, "A colliding node should return the existing node"


def test_circular_linked_list_str():
    head = Node(1)
    cll = CircularLinkedList(head)
    for x in range(2, 5):
        cll.insert_node(Node(x))
    assert str(cll) == '1 -> 2 -> 3 -> 4 -> 1'


def test_cll_iterator():
    head = Node(1)
    cll = CircularLinkedList(head)
    nodes = [Node(x) for x in range(2, 6)]
    for node in nodes:
        cll.insert_node(node)
    nodes = [head, *nodes]
    temp_nodes = []
    for node in cll:
        temp_nodes.append(node)
    assert nodes == temp_nodes, "For more reasource on how to create an iterator: " \
                                "https://www.programiz.com/python-programming/iterator"


def test_cll_traverse_from():
    cll = CLLFactory(node_ids=[100000, 1000000, 10000000, 100000000, 1000000000])
    node = cll.traverse_from(hash_key=False)
    assert node._id == 1000000000
    node = cll.traverse_from(steps=2, hash_key=False)
    assert node._id == 10000000
    node = cll.traverse_from(1000000, steps=7, hash_key=False)
    assert node._id == 100000000
    node = cll.traverse_from(200000, hash_key=False)
    assert node._id == 1000000000
    key = "xyz"
    node = cll.traverse_from(key, steps=2)
    assert node._id == 1000000


# test Naive Distributed Hash Table

def test_ndht_repr():
    dht = NaiveDistributedHashTable(Node(1))
    assert repr(dht) == f"<NaiveDistributedHashTable(head=<Node(id=1, k={k_bit})>, k={k_bit}, size=1)>", f"{repr(dht)}"


def test_ndht_repr_only_uses_cll_repr():
    source = inspect.getsource(NaiveDistributedHashTable)
    assert "__repr__" not in source, "Try making your CircularLinkedList __repr__ method dynamic rather than " \
                                     "overloading the __repr__ method in the subclass. This is not worth many " \
                                     "points and will not be used later but will show you understand inheritance and " \
                                     "how to make your function dynamic."


def test_ndht_get_method():
    node_ids = [100000, 1000000, 10000000, 100000000, 1000000000]
    dht = CLLFactory(cls=NaiveDistributedHashTable, node_ids=node_ids)
    key = "seb@jedi.com"
    data = {'class': 'Real-Time Intelligent Systems'}
    found_node = dht.find_predecessor_node(key)
    found_node.next.db[key] = data
    found_data = dht.get(key)
    assert data == found_data
    dht = CLLFactory(cls=NaiveDistributedHashTable, k=3, node_count=8)
    expected_node = dht.find_successor_node(key)
    expected_node.db[key] = data
    found_data = dht.get(key)
    assert expected_node.db[key] == found_data, "A node with id equal to a hash_key is responsible for the key"


def test_ndht_put_method():
    node_ids = [100000, 1000000, 10000000, 100000000, 1000000000]
    dht = CLLFactory(cls=NaiveDistributedHashTable, node_ids=node_ids)
    key = "seb@jedi.com"
    data = {'class': 'Real-Time Intelligent Systems'}
    node = dht.put(key, data)
    the_node = dht.find_successor_node(100000, hash_key=False)
    assert node == the_node, "The nodes are not the same"
    assert the_node.db == {key: data}, "The stored data is not the same"
    dht = CLLFactory(cls=NaiveDistributedHashTable, k=3, node_count=3)
    node = dht.put(key, data)
    the_node = dht.find_successor_node(4, hash_key=False)
    assert node == the_node, "The nodes are not the same"
    assert the_node.db == {key: data}, "The stored data is not the same"


# test Smart Distributed Hash Table

def test_sdht_repr():
    dht = SmartDistributedHashTable(Node(1))
    assert repr(dht) == f"<SmartDistributedHashTable(head=<Node(id=1, k={k_bit})>, k={k_bit}, size=1)>", f"{repr(dht)}"


def test_sdht_repr_only_uses_cll_repr():
    source = inspect.getsource(SmartDistributedHashTable)
    assert "__repr__" not in source, "Try making your CircularLinkedList __repr__ method dynamic rather than " \
                                     "overloading the __repr__ method in the subclass. This is not worth many " \
                                     "points and will not be used later but will show you understand inheritance and " \
                                     "how to make your function dynamic."


def test_sdht_update_finger_tables():
    dht = CLLFactory(cls=SmartDistributedHashTable, k=3, node_ids=[0, 1, 2, 6])
    dht.update_finger_tables()
    node2 = dht.find_successor_node(2, hash_key=False)
    node6 = dht.find_successor_node(6, hash_key=False)
    node2_expected_finger_table = [(0, 3, node6), (1, 4, node6), (2, 6, node6)]
    assert node2.finger_table == node2_expected_finger_table, "Your finger table is wrong"


def test_sdht_large_finger_table_print():
    dht = CLLFactory(cls=SmartDistributedHashTable, node_count=10)
    dht.update_finger_tables()
    dht.head.print_finger_table()


def test_find_node_using_finger_table():
    n2 = Node(2, k=3)
    dht = SmartDistributedHashTable(n2, k=3)
    n1 = Node(1, k=3)
    n0 = Node(0, k=3)
    n6 = Node(6, k=3)
    for node in [n1, n0, n6]:
        dht.insert_node(node)
    dht.update_finger_tables()
    node = dht.find_node(2, hash_key=False)
    assert node is n2, "Node is not n2"
    node = dht.find_node(3, hash_key=False)
    assert node is n6, "Node is not n6"
    dht.head = n1
    node = dht.find_node(7, hash_key=False)
    assert node is n0, "Node is not n0"


def test_sdht_get_method():
    dht = CLLFactory(cls=SmartDistributedHashTable, k=3, node_ids=[1, 2, 6, 0])
    dht.update_finger_tables()
    key = "seb@yahoo.com"
    data = {'class': 'Real-Time Intelligent Systems'}
    found_node = dht.find_node(key)
    found_data = dht.get(key)
    assert found_data is None, "There should be no data for the key in the responsible node"
    found_node.db[key] = data
    found_data = dht.get(key)
    assert found_data == data, "The data is not the same"


def test_sdht_put_method():
    dht = CLLFactory(cls=SmartDistributedHashTable, k=3, node_ids=[1, 2, 6, 0])
    dht.update_finger_tables()
    key = "seb@jedi.com"
    key_hash = c_hash(key, k=3)
    data = {'class': 'Real-Time Intelligent Systems'}
    node = dht.put(key, data)
    assert node._id == 6


if __name__ == '__main__':
    test_name = input().strip()
    try:
        globals()[test_name]()
        print("Passed")
    except Exception as e:
        print("Failed")
        print(e)