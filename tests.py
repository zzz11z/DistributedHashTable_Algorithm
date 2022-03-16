import inspect
import re
import random
import argparse

from test_runner import TestRunner as TR

from solution import (k_bit, distance, question_1, c_hash, Node, CircularLinkedList, NaiveDistributedHashTable,
                               SmartDistributedHashTable, CLLFactory, ArgumentError)

random.seed(123)


@TR.points(10)
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

@TR.points(10)
def test_distance_function():
    error_msg = "Try again - distance(%d, %d)"
    assert distance(0, 100) == 100, error_msg % (0, 100)
    assert distance(100, 1000) == 900, error_msg % (100, 1000)
    assert distance(4e9, 100) == 294967396, error_msg % (4e9, 100)
    assert distance(1, 5, 3) == 4, error_msg % (1, 5)
    assert distance(5, 3, 3) == 6, error_msg % (5, 3)


@TR.points(5)
def test_question_1():
    print(question_1(), end="")


@TR.points(15)
def test_consistent_hashing_algorithm():
    hash1 = c_hash("john@gmail.com")
    hash2 = c_hash("john@gmail.com")
    hash3 = c_hash("jane@yahoo.com")
    assert hash1 == hash2, "hash algorithm is unstable. Hashes of the same value must be equal"
    assert isinstance(hash1, int), "hashing algorithm must return an int"
    assert hash1 <= 2**k_bit, "hashing algorithm must implement consistent hashing"
    assert hash1 != hash3, "hashes should be different with high probability"
    hash1 = c_hash("joe@uchicago.edu", k=3)
    assert hash1 <= 2**3, "hashing algorithm must implement consistent hashing to k=3"


# test naive node

@TR.points(5)
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


@TR.points(2)
def test_node_repr():
    node = Node(random.getrandbits(k_bit))
    assert repr(node) == f"<Node(id={node._id:,}, k={k_bit})>"


@TR.points(2)
def test_node_str():
    node = Node(random.getrandbits(k_bit))
    assert str(node) == f"{node._id}", "You didn't define the node __str__ function correctly"


@TR.points(5)
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


@TR.points(10)
def test_node_print_finger_table():
    node = Node(4, k=32)
    for i in range(3):
        node.finger_table.append((i, (node._id + 2**i) % 2**3, node))
    node.print_finger_table()


# test circular linked list

@TR.points(5)
def test_circular_linked_list_constructor():
    node = Node(random.getrandbits(k_bit))
    cll = CircularLinkedList(node)
    assert node == cll.head, "The passed node should be the head"
    assert cll.head == cll.head.next, "head of circular liked list with one node should point to itself"
    assert cll.size == 1, "The size of newly constructed circular linked list should be 1"
    assert cll.k == k_bit
    cll = CircularLinkedList(Node(1), k=3)
    assert cll.k == 3


@TR.points(5)
def test_circular_linked_list_repr():
    id_val = random.getrandbits(k_bit)
    cll = CircularLinkedList(Node(id_val))
    assert repr(cll) == f"<CircularLinkedList(" \
                        f"head=<Node(id={id_val:,}, k={k_bit})>, k={k_bit}, size={cll.size})>", f"{repr(cll)}"


@TR.points(10)
def test_circular_linked_list_find_predecessor_node():
    head = Node(random.getrandbits(k_bit))
    cll = CircularLinkedList(head)
    rand_key = random.getrandbits(k_bit)
    node = cll.find_predecessor_node(rand_key, hash_key=False)
    assert node is head, "The predecessor node to key of a single node circular linked list is the head"
    nodes = []
    current = head
    for x in range(4):
        node = Node(head._id + int(1e8*(x+1)))
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


@TR.points(5)
def test_circular_linked_list_find_successor_node():
    head = Node(100000)
    cll = CircularLinkedList(head)
    nodes = []
    current = head
    for x in range(4):
        node = Node(head._id + int(1e8*(x+1)))
        nodes.append(node)
        current.set_next(node)
        current = node
    current.set_next(head)
    check_key = int((nodes[2]._id + nodes[3]._id) / 2)
    node = cll.find_successor_node(check_key, hash_key=False)
    assert node is nodes[3]


@TR.points(10)
def test_circular_linked_list_insert_node():
    head = Node(int(1e3))
    cll = CircularLinkedList(head)
    n1 = cll.insert_node(Node(int(1e7)))
    n2 = cll.insert_node(Node(int(1e4)))
    n3 = cll.insert_node(Node(1))
    assert head.next is n2, "head.next"
    assert n2.next is n1, "n2.next"
    assert n1.next is n3, "n1.next"
    assert n3.next is head, "n3.next"
    assert cll.size == 4, "size"
    n4 = cll.insert_node(Node(1))
    assert n4 is n3, "A colliding node should return the existing node"


@TR.points(5)
def test_cll_sorted_insert_refactored():
    for func in (CircularLinkedList.insert_node, CircularLinkedList.find_predecessor_node):
        source = inspect.getsource(func)
        func_docs = re.search(r"\"{3}.+\"{3}", source, flags=re.DOTALL)
        solution_line_count = source.count('\n') - func_docs.group().count('\n')
        assert solution_line_count <= 10, "If you're clever and write really logical code, you can" \
                                          "get these functions to 10 lines or less. But don't worry if you can't. " \
                                          "This question isn't worth much, and isn't necessary to pass other" \
                                          "test cases."


@TR.points(5)
def test_circular_linked_list_str():
    head = Node(1)
    cll = CircularLinkedList(head)
    for x in range(2, 5):
        cll.insert_node(Node(x))
    assert str(cll) == '1 -> 2 -> 3 -> 4 -> 1'


@TR.points(15)
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
    assert nodes == temp_nodes, "For more resources on how to create an iterator: " \
                                "https://www.programiz.com/python-programming/iterator"


@TR.points(10)
def test_cll_traverse_from():
    cll = CLLFactory(node_ids=[100000, 1000000, 10000000, 100000000, 1000000000])
    node = cll.traverse_from(hash_key=False)
    assert node._id == 1000000000, "case 1"
    node = cll.traverse_from(steps=2, hash_key=False)
    assert node._id == 10000000, "case 2"
    node = cll.traverse_from(1000000, steps=7, hash_key=False)
    assert node._id == 100000000, "case 3"
    node = cll.traverse_from(200000, hash_key=False)
    assert node._id == 1000000000,"case 4"
    key = "xyz"
    node = cll.traverse_from(key, steps=2)
    assert node._id == 1000000, "case 5"


# test Naive Distributed Hash Table

@TR.points(2)
def test_ndht_repr():
    dht = NaiveDistributedHashTable(Node(1))
    assert repr(dht) == f"<NaiveDistributedHashTable(head=<Node(id=1, k={k_bit})>, k={k_bit}, size=1)>", f"{repr(dht)}"


@TR.points(2)
def test_ndht_repr_only_uses_cll_repr():
    source = inspect.getsource(NaiveDistributedHashTable)
    assert "__repr__" not in source, "Try making your CircularLinkedList __repr__ method dynamic rather than " \
                                     "overloading the __repr__ method in the subclass. This is not worth many " \
                                     "points and will not be used later but will show you understand inheritance and " \
                                     "how to make your function dynamic."


@TR.points(10)
def test_ndht_get_method():
    node_ids = [100000, 1000000, 10000000, 100000000, 1000000000]
    dht = CLLFactory(cls=NaiveDistributedHashTable, node_ids=node_ids)
    key = "seb@jedi.com"
    data = {'class': 'Real-Time Intelligent Systems'}
    found_node = dht.find_predecessor_node(key)
    found_node.next.db[key] = data
    found_data = dht.get(key)
    assert data == found_data, f"found_data is {found_data}"
    dht = CLLFactory(cls=NaiveDistributedHashTable, k=3, node_count=8)
    expected_node = dht.find_successor_node(key)
    expected_node.db[key] = data
    found_data = dht.get(key)
    assert expected_node.db[key] == found_data, "A node with id equal to a hash_key is responsible for the key"



@TR.points(10)
def test_ndht_put_method():
    node_ids = [100000, 1000000, 10000000, 100000000, 1000000000]
    dht = CLLFactory(cls=NaiveDistributedHashTable, node_ids=node_ids)
    key = "seb@jedi.com"
    data = {'class': 'Real-Time Intelligent Systems'}
    node = dht.put(key, data)
    the_node = dht.find_successor_node(100000, hash_key=False)
    assert node == the_node, "The nodes are not the same 1"
    assert the_node.db == {key: data}, "The stored data is not the same"
    dht = CLLFactory(cls=NaiveDistributedHashTable, k=3, node_count=3)
    node = dht.put(key, data)
    the_node = dht.find_successor_node(4, hash_key=False)
    assert node == the_node, "The nodes are not the same 2"
    assert the_node.db == {key: data}, "The stored data is not the same"



# test Smart Distributed Hash Table

@TR.points(2)
def test_sdht_repr():
    dht = SmartDistributedHashTable(Node(1))
    assert repr(dht) == f"<SmartDistributedHashTable(head=<Node(id=1, k={k_bit})>, k={k_bit}, size=1)>", f"{repr(dht)}"


@TR.points(2)
def test_sdht_repr_only_uses_cll_repr():
    source = inspect.getsource(SmartDistributedHashTable)
    assert "__repr__" not in source, "Try making your CircularLinkedList __repr__ method dynamic rather than " \
                                     "overloading the __repr__ method in the subclass. This is not worth many " \
                                     "points and will not be used later but will show you understand inheritance and " \
                                     "how to make your function dynamic."


@TR.points(10)
def test_sdht_update_finger_tables():
    dht = CLLFactory(cls=SmartDistributedHashTable, k=3, node_ids=[0, 1, 2, 6])
    dht.update_finger_tables()
    node2 = dht.find_successor_node(2, hash_key=False)
    node6 = dht.find_successor_node(6, hash_key=False)
    node2_expected_finger_table = [(0, 3, node6), (1, 4, node6), (2, 6, node6)]
    assert node2.finger_table == node2_expected_finger_table, "Your finger table is wrong"


@TR.points(15)
def test_sdht_large_finger_table():
    dht = CLLFactory(cls=SmartDistributedHashTable, node_count=10)
    dht.update_finger_tables()
    dht.head.print_finger_table()


@TR.points(20)
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


@TR.points(10)
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



@TR.points(10)
def test_sdht_put_method():
    dht = CLLFactory(cls=SmartDistributedHashTable, k=3, node_ids=[1, 2, 6, 0])
    dht.update_finger_tables()
    key = "seb@jedi.com"
    key_hash = c_hash(key, k=3)
    data = {'class': 'Real-Time Intelligent Systems'}
    node = dht.put(key, data)
    assert node._id == 6


# your turn to test

@TR.points(20)
def test_remove_node():
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
    assert node0.db[key] == data, "Your db is wrong"





mytest = "test_remove_node"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output.')
    parser.add_argument('-ff', '--fail_fast', action='store_true',
                        help="Halt test runner upon first fail.")
    parser.add_argument('-sp', '--enable_print', action='store_true',
                        help="Suppress print for the test functions that are print based.")
    args = parser.parse_args()

    verbose = args.verbose
    fail_fast = args.fail_fast
    suppress_print = not args.enable_print

    #runner = TR(verbose=verbose, fail_fast=fail_fast, suppress_print=suppress_print)
    #runner.run()
    runner = TR()
    isolated_tests = ("test_circular_linked_list_find_predecessor_node","test_circular_linked_list_find_successor_node",mytest)
    runner.run_tests(*isolated_tests)
