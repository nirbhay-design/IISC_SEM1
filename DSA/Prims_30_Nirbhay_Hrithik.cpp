#include <iostream>
#include <climits>
using namespace std;

struct MinHeapNode {
    int vertex;
    int key;
};

class MinHeap {
private:
    MinHeapNode* heap;
    int* pos;
    int size;
    int capacity;

public:
    MinHeap(int capacity) {
        this->capacity = capacity;
        heap = new MinHeapNode[capacity];
        pos = new int[capacity];
        size = 0;

        for (int i = 0; i < capacity; i++) {
            pos[i] = -1;
        }
    }

    void insert(int vertex, int key) {
        heap[size] = {vertex, key};
        pos[vertex] = size;
        size++;
        decreaseKey(vertex, key);
    }

    bool isEmpty() {
        return size == 0;
    }

    int extractMin() {
        if (isEmpty()) return -1;

        int root = heap[0].vertex;

        heap[0] = heap[size - 1];
        pos[heap[0].vertex] = 0;

        size--;
        heapifyDown(0);

        return root;
    }

    void decreaseKey(int vertex, int key) {
        int i = pos[vertex];
        heap[i].key = key;

        while (i != 0 && heap[i].key < heap[(i - 1) / 2].key) {
            swap(&heap[i], &heap[(i - 1) / 2]);
            i = (i - 1) / 2;
        }
    }

    bool isInHeap(int vertex) {
        return pos[vertex] < size;
    }

private:
    void swap(MinHeapNode* x, MinHeapNode* y) {
        MinHeapNode temp = *x;
        *x = *y;
        *y = temp;

        pos[x->vertex] = (x - heap);
        pos[y->vertex] = (y - heap);
    }

    void heapifyDown(int i) {
        int smallest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (left < size && heap[left].key < heap[smallest].key)
            smallest = left;
        if (right < size && heap[right].key < heap[smallest].key)
            smallest = right;

        if (smallest != i) {
            swap(&heap[i], &heap[smallest]);
            heapifyDown(smallest);
        }
    }
};

class Prims {
private:
    int vertex;
    int** adjMatrix;

public:
    Prims(int n) {
        vertex = n;
        adjMatrix = new int*[n];
        for (int i = 0; i < n; i++) {
            adjMatrix[i] = new int[n];
            for (int j = 0; j < n; j++) {
                adjMatrix[i][j] = INT_MAX;
            }
        }
    }

    void input() {
        int edges;
        cout << "Enter the number of edges: ";
        cin >> edges;
        if (edges < vertex - 1){
            cout << "Edges too low in number!";
            exit(0);
        } 

        for (int i = 0; i < edges; i++) {
            int u, v, w;
            cout << "enter edge (u, v, w) " << i + 1 << ": ";  
            cin >> u >> v >> w;
            adjMatrix[u][v] = w;
            adjMatrix[v][u] = w;
        }
    }

    void get_MST() {
        int* key = new int[vertex];
        int* parent = new int[vertex];
        bool* inMST = new bool[vertex];

        for (int i = 0; i < vertex; i++) {
            key[i] = INT_MAX;
            parent[i] = -1;
            inMST[i] = false;
        }

        MinHeap minHeap(vertex);

        key[0] = 0;
        minHeap.insert(0, 0);

        while (!minHeap.isEmpty()) {
            int u = minHeap.extractMin();
            inMST[u] = true;
            for (int v = 0; v < vertex; v++) {
                if (adjMatrix[u][v] != INT_MAX && !inMST[v] && adjMatrix[u][v] < key[v]) {
                    key[v] = adjMatrix[u][v];
                    parent[v] = u;

                    if (minHeap.isInHeap(v)) {
                        minHeap.decreaseKey(v, key[v]);
                    }
                    
                    minHeap.insert(v, key[v]);
                    
                }
            }
        }

        cout << "Minimum Spanning Tree edges:\n";
        int minCost = 0;
        for (int i = 1; i < vertex; i++) {
            cout << "Edge: (" << parent[i] << ", " << i << ") Weight: " << adjMatrix[i][parent[i]] << "\n";
            minCost += adjMatrix[i][parent[i]];
        }
        cout << "Total cost of MST: " << minCost << "\n";

    }
};

int main() {
    int n;
    cout << "Enter the number of vertices: ";
    cin >> n;

    Prims p(n);
    p.input();
    p.get_MST();

    return 0;
}