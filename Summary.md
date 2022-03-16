
# Summary

This homework covers distributed hash tables which is a way for us to store data across a scalable network of computers
which allows for an arbitrarily large database. We, of course, don't have the resources to create and test a system 
across multiple machines, so instead we will be simulating our network of machines using a circular linked list.

### Circular Linked List

A circular linked list is a special kind of linked list that consists of nodes that each point to a single successor
node. We can talk about the successor node as the rightmost neighbor or the closest clockwise neighbor. A simple 
singular linked list has a head node, a chain of body nodes, and tail node which points to nothing. A circular
linked list has the special property that the tail node points to the head node.

    ----------    ----------             ----------    
    |  HEAD  | -> | Node i | ->  ...  -> |  HEAD  |
    ----------    ----------             ----------

In our distributed hash table, each node will not only contain a pointer to its successor node, but will also contain
a storage container for our data, which we will represent as a python dictionary. It will also contain a finger table
which will be discussed in further detail below.

### Consistent Hashing

For a detailed explanation of how hashing and python dictionaries work, see <br>
["Python behind the scenes #10: how Python dictionaries work"](https://tenthousandmeters.com/blog/python-behind-the-scenes-10-how-python-dictionaries-work/)

The problem with ordinary hash tables comes when we need to rehash a key. Suppose we have k records and N machines. On
average, each machine is responsible for k/N records and the function which determines which node stores record i is:

    index = hash(key) % N

Where index is the id of the machine responsible for the key. If machine j leaves the network then N becomes N-1 and
not only will we need to determine who now owns records previously owned by machine j, but additionally it is highly
probable that many records owned by machines j will also need to be redistributed. Consistent hashing solves this
problem. 

With consistent hashing we toss the dependency on the number of servers and replace it with a circle of fixed size 
where record responsibility is determined by dividing the circle into slices and designating representative (machine) 
for each slice. If a representative leaves, rather than having to reassign every record, a neighboring representative
assumes the responsibility. In that way, a remapping will only require k/N reassignments.

We will of course need a strong and consistent way to hash a key. For our purposes, we will use the SHA1 cryptographic
hash function.

See ["Consistent Hashing and Random Trees - Karger et al. - MIT"](https://people.csail.mit.edu/karger/Papers/web.pdf)

### Distributed Hash Table

For our purposes, we will create a series of Nodes each with a k bit id, and connect them using our circular linked
list. These nodes should be inserted in order of their id where the node with the maximum id connects to the head.

A node, M, will assume responsibility for all records where:

    predecessor_id < record_key <= M_id 

For example, given a node with id 2 and a node with id 8, every key that gets mapped to 3, 4, 5, 6, 7, or 8 will be
"managed" by node 8.


### Finger Tables / Chord Protocol

When we only have a few machines in our system, reads, writes, and deletion of records will be fairly fast given 
that finding the responsible node only takes a few hops. But when the number of machines balloons into the thousands, 
The number of hops coupled with network latency can significantly increase the time waiting on an operation to 
complete. Finger tables solve this problem.

In addition to knowing it's successor neighbor, a node will also have a finger table it which will include k entries, 
where k is the number of bits used in our consistent hashing scheme. Each entry will store the successor hash given by:

    (id + 2**i) % 2**k

Where "id" is the id of the node in question, and i is an integer between 0 and k-1, inclusive. Each entry will also
store a reference to the successor node to the successor hash. 

To implement a finger table lookup, give a query, a node will first determine if its or its successor is responsible,
and, if not, will forward the query to the greatest successor node in the finger table not exceeding the key hash. This
will continue until the responsible node is found. 

See ["Chord: A Scalable Peer-to-peer Lookup Service for Internet
Applications"](https://pdos.csail.mit.edu/papers/chord:sigcomm01/chord_sigcomm.pdf) for more details.