class Queue {

    elements: any;
    head: number;
    tail: number;

    constructor() {
        this.elements = {};
        this.head = 0;
        this.tail = 0;
    }

    enqueue(element: any) {
        this.elements[this.tail] = element;
        this.tail++;
    }

    dequeue() {
        const item = this.elements[this.head];
        delete this.elements[this.head];
        this.head++;
        return item;
    }


    get length() {
        return this.tail - this.head;
    }


}

export default Queue;