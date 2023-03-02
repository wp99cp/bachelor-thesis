import {Component, OnInit} from '@angular/core';

enum Classes {
    Background = 0,
    Snow = 1,
    Clouds = 2,
    Water = 3
}

const Background_Color = [255, 255, 255]
const Snow_Color = [38, 211, 192]
const Clouds_Color = [192, 38, 211]
const Water_Color = [211, 192, 38]
const Class_Colors = [Background_Color, Snow_Color, Clouds_Color, Water_Color]

const SCENE_CODES = ['original', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

const BASE_URL = 'http://192.168.2.38:5000';

@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
    public current_color = Classes.Clouds;

    // order of colors is important
    private canvas_history: ImageData[] = []
    private current_image_ref: string | null = null;
    public loading_next: boolean = false;


    private enter_is_down: boolean = false;
    private control_is_down: boolean = false;


    private color_adding_mode: boolean = false;
    private color_adding_mode_disabled: boolean = false;
    private active_scene_code: string = 'scene_original';
    public threshold: number = 12;

    ngOnInit(): void {


        this.next_img(true)

        const canvas = document.querySelectorAll('canvas')[0]
        const context_or_null: CanvasRenderingContext2D | null = canvas.getContext('2d')

        if (context_or_null === null) throw new Error('context_or_null is null')
        const context = context_or_null as CanvasRenderingContext2D


        let lineWidth = 0
        let isMousedown = false
        let points: { x: number; y: number; lineWidth: number; }[] = []

        // set shortcut listeners for actions
        document.addEventListener('keydown', (e: KeyboardEvent) => {

            // undo
            if (e.key === '-') {
                this.undoDraw();
            }

            if (e.key === 'Enter') {
                this.enter_is_down = true;
            }

            if (e.key === '+') {
                this.color_adding_mode = true;
                this.color_adding_mode_disabled = true;
            }

            if (e.key === 'Control') {
                this.control_is_down = true;
            }

            if (e.key === 'ArrowUp') {
                this.threshold += 1;
            }

            if (e.key === 'ArrowDown') {
                this.threshold -= 1;
            }


            if (this.control_is_down && this.enter_is_down) {
                this.next_img()
            }

            if (e.key === '/') {
                this.clear()
            }

        });

        // add shortcut listeners for band switching
        // those listeners must be pressed, once released, the band will be switched back to the original one
        document.addEventListener('keydown', (e: KeyboardEvent) => {

            // if any number is pressed, switch to the corresponding band
            if (SCENE_CODES.includes(e.key)) {
                this.switch_to_alternative(e.key);
            }

            if (e.key === '.') {

                // hide annotations
                const canvas_id = 'canvas';
                const canvas = document.getElementById(canvas_id)
                canvas!.style.opacity = '0';


            }

        });

        document.addEventListener('keyup', (e: KeyboardEvent) => {

            // if any number is pressed, switch to the corresponding band
            if (SCENE_CODES.includes(e.key)) {
                this.switch_to_original();
            }

            if (e.key === '.') {

                // hide annotations
                const canvas_id = 'canvas';
                const canvas = document.getElementById(canvas_id)
                canvas!.style.opacity = '1';

            }

            if (e.key === 'Enter') {
                this.enter_is_down = false;
            }

            if (e.key === '+') {
                this.color_adding_mode = false;
            }

            if (e.key === 'Control') {
                this.control_is_down = false;
            }

        });


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

                if (this.color_adding_mode) {
                    points = [];
                    isMousedown = false
                    return;
                }


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
                    if (data[i] === Background_Color[0] && data[i + 1] === Background_Color[1] && data[i + 2] === Background_Color[2]) {
                        data[i + 3] = 0
                    }
                }
                context.putImageData(imageData, 0, 0)

                this.canvas_history.push(context.getImageData(0, 0, canvas.width, canvas.height))
                this.get_coords(e, canvas);

                isMousedown = false

                requestIdleCallback(() => points = []);

                lineWidth = 0
            })
        }


        this.switch_to_original();


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

        if (this.color_adding_mode) {
            this.add_color_under_cursor(context, stroke);
            return;
        }

        const color = Class_Colors[this.current_color]

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

    switch_to_alternative(scene_code: string) {

        this.active_scene_code = 'scene_' + scene_code

        const scene_codes = JSON.parse(JSON.stringify(SCENE_CODES));

        // remove current scene from list
        const index = scene_codes.indexOf(scene_code);
        if (index > -1) scene_codes.splice(index, 1);
        else throw new Error('scene code not found')

        // hide all scenes
        for (const code of scene_codes) {
            const scene = document.getElementById('scene_' + code)
            scene!.style.display = 'none'
        }

        const scene = document.getElementById('scene_' + scene_code)
        scene!.style.display = 'block'

    }

    switch_to_original() {
        this.active_scene_code = 'scene_original'
        this.switch_to_alternative('original');
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

        if (!first_time && this.current_image_ref !== null) {


            const dataURL = canvas.toDataURL('image/png')
            const img = new Image()
            img.onload = () => {
                context.drawImage(img, 0, 0)
            }


            // send the image as a post to the backend
            const xhr = new XMLHttpRequest()
            xhr.open('POST', BASE_URL + '/update_mask/' + this.current_image_ref)
            xhr.setRequestHeader('Content-Type', 'image/png')
            xhr.send(JSON.stringify({image: dataURL}))

        }

        // clear the canvas and clear the history
        context.clearRect(0, 0, canvas.width, canvas.height);
        this.canvas_history = [];


        setTimeout(() => {

            // make a request to the backend to get the image
            // backend is at port 5000
            const url = BASE_URL + '/next_image'
            fetch(url)
                .then(response => response.json())
                .then(json => {

                    const scenes = json['scenes'];
                    console.log(scenes);
                    this.current_image_ref = json['ref'];

                    const base_path = BASE_URL + '/imgs/';

                    // Load all scenes
                    for (const scene_code of SCENE_CODES) {
                        console.log('loading scene scene_' + scene_code);
                        (document.getElementById('scene_' + scene_code) as HTMLImageElement)!.src = base_path + scenes['scene_' + scene_code];
                    }

                    this.switch_to_original();
                    this.loading_next = false;
                });


        }, 250);

    }


    private async add_color_under_cursor(context: CanvasRenderingContext2D, stroke: string | any[]) {

        const current_position = stroke[stroke.length - 1];

        this.color_adding_mode_disabled = true;

        const image = document.getElementById(this.active_scene_code) as HTMLImageElement;
        const marked_pixels: any = await this.find_similar_pixels(image, current_position.x, current_position.y);
        5
        const color = Class_Colors[this.current_color]
        context.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`
        context.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`

        console.log("Similar pixels computed: ", marked_pixels.length)

        marked_pixels.forEach((pixel: any) => {
            context.fillRect(pixel.x, pixel.y, 1, 1);
        });

        this.canvas_history.push(context.getImageData(0, 0,1024, 1024))
        console.log("Color added")

    }

    private find_similar_pixels(image: HTMLImageElement, x: number, y: number) {

        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 1024;

        const ctx = canvas.getContext('2d');

        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.src = image.src;


        return new Promise((resolve) => {
            const threshold = this.threshold ** 2;
            const imageData = new Uint8ClampedArray(canvas.width * canvas.height * 4);
            const visited = new Uint8Array(canvas.width * canvas.height);
            const similar_pixels: any = [];

            img.onload = () => {
                ctx?.drawImage(img, 0, 0, canvas.width, canvas.height);
                const color = ctx?.getImageData(x, y, 1, 1).data.slice(0, 3);
                if (!color) throw new Error('color is null')

                // Copy the pixel data from the canvas into the imageData array
                const canvasData = ctx?.getImageData(0, 0, canvas.width, canvas.height).data;
                if (!canvasData) throw new Error('canvasData is null')
                for (let i = 0; i < canvasData.length; i++) {
                    imageData[i] = canvasData[i];
                }

                const queue = [{x, y}];
                while (queue.length) {
                    const {x, y} = queue.shift()!;
                    const index = (y * canvas.width + x) * 4;

                    // Check if pixel is within bounds and hasn't been visited
                    if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height || visited[index / 4]) {
                        continue;
                    }

                    // Check if pixel color is similar to the target color
                    const pixelColor = imageData.subarray(index, index + 3);
                    if (Math.abs(pixelColor[0] - color[0]) + Math.abs(pixelColor[1] - color[1]) + Math.abs(pixelColor[2] - color[2]) > threshold) {
                        continue;
                    }

                    // Add pixel to similar_pixels list
                    similar_pixels.push({x, y});

                    // Mark pixel as visited
                    visited[index / 4] = 1;

                    // Add adjacent pixels to queue
                    queue.push({x: x - 1, y}); // left
                    queue.push({x: x + 1, y}); // right
                    queue.push({x, y: y - 1}); // top
                    queue.push({x, y: y + 1}); // bottom
                }

                resolve(similar_pixels);
            };
        });


    }

}
