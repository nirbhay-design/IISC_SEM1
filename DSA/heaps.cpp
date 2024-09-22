#include <bits/stdc++.h>

using namespace std; 

class maxHeap{
    public:
    int heap[10000];
    int hlen = 0;

    void maxHeapify(int index) {
        int left = 2 * index; 
        int right = left + 1;
        int largest = -1;
        if (left <= hlen && heap[index] < heap[left]){
            largest = left;
        } else {
            largest = index;
        }

        if (right <= hlen && heap[largest] < heap[right]) {
            largest = right;
        }

        if (largest != index) {
            int temp = heap[index];
            heap[index] = heap[largest];
            heap[largest] = temp;
            maxHeapify(largest);
        }

    }

    void printHeap() {
        for (int i = 1; i<= hlen; i++) {
            cout << heap[i] << " ";
        }
        cout << "\n";
    }
    

    void copytoHeap(int * A, int n){
        hlen = n;
        for (int i = 0; i < hlen; i++) {
            heap[i+1] = A[i];
        }
    }

    void buildMaxHeap(int * A, int n) {
        copytoHeap(A, n);

        for (int i = hlen / 2; i > 0; i--) {
            maxHeapify(i);
        }
    }

    int extractMaxElement() {
        if (hlen == 0) {
            cout << "heap underflow\n";
            return INT_MIN;
        }

        return heap[1];
    }

    void deleteFromMaxHeap() {
        if (hlen == 0) {
            cout << "heap underflow" << endl;
            return;
        }
        heap[1] = heap[hlen];
        hlen--;
        maxHeapify(1);
    }

    void insert(int element) {
        heap[++hlen] = element;
        int idx = hlen;
        int pi = idx / 2;
        while (pi >= 1 && heap[pi] < heap[idx]) {
            int temp = heap[pi];
            heap[pi] = heap[idx];
            heap[idx] = temp;
            idx = pi;
            pi = idx / 2;
        }
    }

    int * sortheap() {
        for (int i = hlen; i > 1; i--) {
            int temp = heap[i];
            heap[i] = heap[1];
            heap[1] = temp;
            hlen--;
            maxHeapify(1);
        }
        return heap;
    }
};


int * heapSort(int * A, int n) {
    maxHeap * mh = new maxHeap();
    mh->buildMaxHeap(A, n);
    return mh->sortheap();
}

void printArr(int * arr, int n){
    for (int i =1;i<=n;i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

int main () {
    maxHeap * mxhp = new maxHeap();

    cout << "0: For performing operations on heap\n1: for sorting an array using heap sort\n";
    while (true) {
        int oper, n;
        cout << "enter your operation[0/1]: ";
        cin >> oper;
        if (oper == 0) {
            cout << "0: build heap from array\n1: insert into the heap\n2: get max element\n3: delete max element\n";
            int heapoper;
            cout << "enter your operation[0/1/2/3]: ";
            cin >> heapoper;
            if (heapoper == 0) {
                cout << "enter number of elements in array: ";
                cin >> n;
                int arr[n];
                for (int i = 0;i<n;i++) {
                    cout << "enter arr[" << i << "]: ";
                    cin >> arr[i];
                }
                mxhp->buildMaxHeap(arr, n);
                mxhp->printHeap();
            } else if (heapoper == 1) {
                int element;
                cout << "enter element to insert: ";
                cin >> element;
                mxhp->insert(element);
                mxhp->printHeap();
            } else if (heapoper == 2) {
                int maxelem = mxhp->extractMaxElement();
                cout << "max element is: " << maxelem << endl;
            } else {
                mxhp->deleteFromMaxHeap();
                mxhp->printHeap();
            }
        } else {
            cout << "enter number of elements in array: ";
            cin >> n;
            int arr[n];
            for (int i = 0;i<n;i++) {
                cout << "enter arr[" << i << "]: ";
                cin >> arr[i];
            }
            int * sortarr = heapSort(arr, n);
            printArr(sortarr, n);
        }
    }
    // int arr[] = {10, 20, 34, 61, 1, 2, 3};
    // int arr[] = {1, 2, 3, 4, 5, 6, 7, 2, 1};
    // int n = sizeof(arr) / sizeof(arr[0]);
    // mxhp->buildMaxHeap(arr, n);
    // mxhp->printHeap();
    // mxhp->insert(56);
    // mxhp->printHeap();
    // mxhp->insert(500);
    // mxhp->printHeap();
    // mxhp->deleteFromMaxHeap();
    // mxhp->printHeap();
    // mxhp->insert(200);
    // mxhp->printHeap();
    // mxhp->deleteFromMaxHeap();
    // mxhp->deleteFromMaxHeap();
    // mxhp->deleteFromMaxHeap();
    // mxhp->deleteFromMaxHeap();
    // mxhp->deleteFromMaxHeap();
    // mxhp->deleteFromMaxHeap();
    // mxhp->deleteFromMaxHeap();
    // mxhp->deleteFromMaxHeap();
    // mxhp->deleteFromMaxHeap();
    // mxhp->deleteFromMaxHeap();
    // mxhp->deleteFromMaxHeap();
    
    
    // mxhp->printHeap();

    // int * sortarr = heapSort(arr, n);
    // printArr(sortarr, n);
    // // mxhp->printHeap();
    return 0;
}