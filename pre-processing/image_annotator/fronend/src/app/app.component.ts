import {Component, OnInit} from '@angular/core';

@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
    alternative_switcher_texts = ['the original image', 'another day\'s image']
    alternative_image_switcher_state = 0;
    public current_color = 1;
    private colors = [[255, 192, 203], [255, 127, 80], [255, 255, 255]]
    private canvas_history: ImageData[] = []
    private current_image_id: string | null = null;
    public loading_next: boolean = false;

    ngOnInit(): void {


        this.next_img(true)

        const canvas = document.querySelectorAll('canvas')[0]
        const context_or_null: CanvasRenderingContext2D | null = canvas.getContext('2d')

        if (context_or_null === null) throw new Error('context_or_null is null')
        const context = context_or_null as CanvasRenderingContext2D


        let lineWidth = 0
        let isMousedown = false
        let points: { x: number; y: number; lineWidth: number; }[] = []

        const requestIdleCallback = window.requestIdleCallback || function (fn: () => void) {
            setTimeout(fn, 1)
        };

        if (!context) throw new Error('context is null')
        context.imageSmoothingEnabled = false

        let path = new Path2D();


        for (const ev of ["touchstart", "mousedown"]) {
            canvas.addEventListener(ev, (e: any) => {

                const {x, y} = this.get_coords(e, canvas)
                isMousedown = true

                lineWidth = 2;
                context.lineWidth = lineWidth// pressure * 50;

                points.push({x, y, lineWidth})
                this.draw_on_canvas(context, points, path);
            })
        }

        for (const ev of ['touchmove', 'mousemove']) {
            canvas.addEventListener(ev, (e: any) => {
                if (!isMousedown) return
                e.preventDefault()

                let {x, y} = this.get_coords(e, canvas)
                lineWidth = 2
                points.push({x, y, lineWidth})
                this.draw_on_canvas(context, points, path);

            })
        }

        for (const ev of ['touchend', 'touchleave', 'mouseup']) {

            canvas.addEventListener(ev, (e: any) => {

                if (points.length > 0) {

                    const fst_p = points[0];
                    const lst_p = points[points.length - 1];

                    const dx = fst_p.x - lst_p.x;
                    const dy = fst_p.y - lst_p.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);

                    if (distance < 25 || this.current_color === 2) {
                        path.closePath();
                        context.fill(path)
                    }

                }

                path = new Path2D();


                // remove white pixels form canvas
                const imageData = context.getImageData(0, 0, canvas.width, canvas.height)
                const data = imageData.data
                for (let i = 0; i < data.length; i += 4) {
                    if (data[i] === this.colors[2][0] && data[i + 1] === this.colors[2][1] && data[i + 2] === this.colors[2][2]) {
                        data[i + 3] = 0
                    }
                }
                context.putImageData(imageData, 0, 0)

                this.canvas_history.push(context.getImageData(0, 0, canvas.width, canvas.height))
                this.get_coords(e, canvas);

                isMousedown = false

                requestIdleCallback(function () {
                    points = []
                })

                lineWidth = 0
            })
        }

    }

    private get_coords(e: any, canvas: HTMLCanvasElement) {

        let canvas_offset_y = canvas.getBoundingClientRect().top
        const canvas_offset_x = canvas.getBoundingClientRect().left

        let x, y;

        if ("touches" in e && e.touches && e.touches[0] && typeof e.touches[0]["force"] !== "undefined") {
            if (e.touches[0]["force"] > 0) {
            }
            x = e.touches[0].pageX
            y = e.touches[0].pageY
        } else {
            x = e.pageX
            y = e.pageY
        }

        x -= canvas_offset_x
        y -= canvas_offset_y

        return {x, y}
    }

    /**
     * This function takes in an array of points and draws them onto the canvas.
     * @param context
     * @param {array} stroke array of points to draw on the canvas
     * @param path
     * @return {void}
     */
    private draw_on_canvas(context: CanvasRenderingContext2D, stroke: string | any[], path: Path2D) {
        const color = this.colors[this.current_color]

        context.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`
        context.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`
        context.lineCap = 'round'
        context.lineJoin = 'round'

        const l = stroke.length - 1
        if (stroke.length >= 3) { // continue line
            const xc = (stroke[l].x + stroke[l - 1].x) / 2
            const yc = (stroke[l].y + stroke[l - 1].y) / 2
            context.lineWidth = stroke[l - 1].lineWidth

            context.quadraticCurveTo(stroke[l - 1].x, stroke[l - 1].y, xc, yc)
            context.stroke()
            context.beginPath()
            context.moveTo(xc, yc)

            path.lineTo(xc, yc)

        } else {  // start a new line
            const point = stroke[l];
            context.lineWidth = point.lineWidth
            context.strokeStyle = point.current_color
            context.beginPath()
            context.moveTo(point.x, point.y)
            context.stroke()
        }
    }

    switch_to_alternative() {

        const alternative = document.getElementById('alternative_image')
        if (alternative!.style.display === 'block') {
            alternative!.style.display = 'none'
            this.alternative_image_switcher_state = 0
        } else {
            alternative!.style.display = 'block'
            this.alternative_image_switcher_state = 1
        }

    }

    clear() {
        const canvas = document.getElementById('canvas') as HTMLCanvasElement
        const context = canvas.getContext('2d')
        if (!context) throw new Error('context is null')

        context.clearRect(0, 0, canvas.width, canvas.height)
    }

    set(c: number) {
        this.current_color = c
    }

    undoDraw() {

        this.canvas_history.pop();
        const canvas = document.getElementById('canvas') as HTMLCanvasElement
        const context = canvas.getContext('2d')
        if (!context) throw new Error('context is null')

        context.clearRect(0, 0, canvas.width, canvas.height)
        context.putImageData(this.canvas_history[this.canvas_history.length - 1], 0, 0)

    }


    next_img(first_time = false) {

        if (this.loading_next) return
        this.loading_next = true;

        // download the canvas as png
        const canvas = document.getElementById('canvas') as HTMLCanvasElement
        const context = canvas.getContext('2d')
        if (!context) throw new Error('context is null')

        if (!first_time && this.current_image_id !== null) {


            const dataURL = canvas.toDataURL('image/png')
            const img = new Image()
            img.onload = () => {
                context.drawImage(img, 0, 0)
            }


            // send the image as a post to the backend
            const xhr = new XMLHttpRequest()
            xhr.open('POST', 'http://192.168.1.223:5000/update_mask/' + this.current_image_id)
            xhr.setRequestHeader('Content-Type', 'image/png')
            xhr.send(JSON.stringify({image: dataURL}))

        }

        // clear the canvas and clear the history
        context.clearRect(0, 0, canvas.width, canvas.height);
        this.canvas_history = [];


        setTimeout(() => {

            // make a request to the backend to get the image
            // backend is at port 5000
            const url = 'http://192.168.1.223:5000/next_image'
            fetch(url)
                .then(response => response.json())
                .then(json => {

                    const base_path = 'http://192.168.1.223:5000/imgs/';
                    (document.getElementById('alternative_image') as HTMLImageElement)!.src = base_path + json['image_alt'];
                    (document.getElementById('image_to_annotate') as HTMLImageElement)!.src = base_path + json['image'];

                    this.current_image_id = json['image'];


                    this.switch_to_alternative();
                    this.switch_to_alternative();

                    this.loading_next = false;
                });


        }, 500);

    }


}
