import {Component, OnInit} from '@angular/core';
import Queue from "./Queue";

enum Classes {
    Background = 0,
    Snow = 1,
    Clouds = 2,
    Water = 3
}


const Background_Color = [255, 255, 255, 0]
const Snow_Color = [38, 211, 192, 255]
const Clouds_Color = [192, 38, 211, 255]
const Water_Color = [211, 192, 38, 255]
const Class_Colors = [Background_Color, Snow_Color, Clouds_Color, Water_Color]
const SCENE_CODES = ['original', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
const BASE_URL = 'http://192.168.2.38:5000';
const RAW_MASK_DIM = 512;

function is_similar_color(pixelColor: Uint8ClampedArray, color: number[], threshold: number) {
    return Math.abs(pixelColor[0] - color[0]) +
        Math.abs(pixelColor[1] - color[1]) +
        Math.abs(pixelColor[2] - color[2]) < threshold

}

function run(gen: () => any, mili: number) {
    const iter = gen();
    const end = Date.now() + mili;
    do {
        const {value, done} = iter.next();
        if (done) return value;
        if (end < Date.now()) {
            console.log("Halted function, took longer than " + mili + " miliseconds");
            return null;
        }
    } while (true);
}


@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
    public current_color = Classes.Clouds;
    // order of colors is important
    private canvas_history: Uint8ClampedArray[] = []
    private current_image_ref: string | null = null;
    public loading_next: boolean = false;
    private canvas_data = Uint8ClampedArray.from({length: RAW_MASK_DIM * RAW_MASK_DIM * 4}, () => 0);
    private enter_is_down: boolean = false;
    private control_is_down: boolean = false;

    private color_adding_mode: boolean = false;
    private active_scene_code: string = 'scene_original';
    public threshold: number = 5;

    private color_adding_mode_disabled: boolean = false;


    ngOnInit(): void {

        this.next_img(true)

        const canvass = document.querySelectorAll('canvas');
        canvass.forEach((canvas: HTMLCanvasElement) => {
            canvas.width = 512
            canvas.height = 512
        });

        let isMousedown = false
        let points: { x: number; y: number }[] = []

        // set shortcut listeners for actions
        document.addEventListener('keydown', (e: KeyboardEvent) => {

            if (e.key === '-') this.undoDraw();
            if (e.key === 'Enter') this.enter_is_down = true;
            if (e.key === 'Control') this.control_is_down = true;
            if (e.key === 'ArrowUp') this.threshold += 1;
            if (e.key === 'ArrowDown') this.threshold -= 1;
            if (e.key === '/') this.clear()

            if (e.key === '+') {
                this.color_adding_mode = true;
            }

            if (this.control_is_down && this.enter_is_down) {
                this.next_img()
            }

        });

        // add shortcut listeners for band switching
        // those listeners must be pressed, once released, the band will be switched back to the original one
        document.addEventListener('keydown', (e: KeyboardEvent) => {

            if (SCENE_CODES.includes(e.key)) this.switch_to_alternative(e.key);

            if (e.key === '.') {
                const canvas = document.querySelectorAll('canvas')
                canvas.forEach((canvas: HTMLCanvasElement) =>  {
                    canvas!.style.opacity = '0';
                });
            }

        });

        document.addEventListener('keyup', (e: KeyboardEvent) => {

            if (SCENE_CODES.includes(e.key)) this.switch_to_original();

            if (e.key === '.') {
                const canvas = document.querySelectorAll('canvas')
                canvas.forEach((canvas: HTMLCanvasElement) =>  {
                    canvas!.style.opacity = '1';
                });
            }

            if (e.key === 'Enter') this.enter_is_down = false;
            if (e.key === 'Control') this.control_is_down = false;

            if (e.key === '+') {
                this.color_adding_mode = false;
                this.color_adding_mode_disabled = true;
            }

        });

        const requestIdleCallback = window.requestIdleCallback || function (fn: () => void) {
            setTimeout(fn, 1)
        };


        let path = new Path2D();

        const canvas = document.getElementById('canvas_write') as HTMLCanvasElement;

        for (const ev of ["touchstart", "mousedown"]) {
            canvas.addEventListener(ev, (e: any) => {

                points = [];
                isMousedown = true

                const {x, y} = this.get_coords(e, canvas)

                points.push({x, y})
                this.draw_on_canvas(points, path);

            })
        }


        for (const ev of ['touchmove', 'mousemove']) {
            canvas.addEventListener(ev, async (e: any) => {


                if (!isMousedown) return
                e.preventDefault()


                if (this.color_adding_mode) {
                    points = [];
                    await this.run_selection(this.get_coords(e, canvas));
                    return;
                }

                let {x, y} = this.get_coords(e, canvas)
                points.push({x, y})
                this.draw_on_canvas(points, path);
            })
        }

        const write_canvas = document.getElementById('canvas_write') as HTMLCanvasElement;
        const context = write_canvas.getContext('2d')! as CanvasRenderingContext2D;

        for (const ev of ['touchend', 'touchleave', 'mouseup']) {

            canvas.addEventListener(ev, async (e: any) => {

                isMousedown = false

                if (this.color_adding_mode) {
                    points = [];
                    await this.run_selection(this.get_coords(e, canvas));
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

                this.merge_and_update_canvases();
                this.canvas_history.push(this.canvas_data)

                isMousedown = false
                requestIdleCallback(() => points = []);


            });
        }

        this.switch_to_original();

    }

    private merge_and_update_canvases() {

        const write_canvas = document.getElementById('canvas_write') as HTMLCanvasElement;
        const write_context = write_canvas.getContext('2d')! as CanvasRenderingContext2D;

        // update canvas_data
        const imageData = write_context.getImageData(0, 0, RAW_MASK_DIM, RAW_MASK_DIM).data;

        // check every pixel
        if (imageData.length !== this.canvas_data.length) throw new Error('imageData and canvas_data have different length');
        for (let i = 0; i < imageData.length; i += 4) {

            // skip transparent pixels
            if (imageData[i + 3] === 0) continue;

            // set correct color
            this.canvas_data[i] = Class_Colors[this.current_color][0];
            this.canvas_data[i + 1] = Class_Colors[this.current_color][1];
            this.canvas_data[i + 2] = Class_Colors[this.current_color][2];
            this.canvas_data[i + 3] = this.current_color == Classes.Background ? 0 : 255;

        }

        // clear write canvas
        write_context.clearRect(0, 0, write_canvas.width, write_canvas.height);

        // update readonly canvas
        const canvas = document.getElementById('canvas_readonly') as HTMLCanvasElement;
        const context = canvas.getContext('2d')! as CanvasRenderingContext2D;
        context.putImageData(new ImageData(this.canvas_data, RAW_MASK_DIM, RAW_MASK_DIM), 0, 0);

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

        const factor = RAW_MASK_DIM / 1024;
        return {"x": Math.floor(x * factor), "y": Math.floor(y * factor)}
    }


    /**
     * This function takes in an array of points and draws them onto the canvas.
     * @param context
     * @param {array} stroke array of points to draw on the canvas
     * @param path
     * @return {void}
     */
    private draw_on_canvas(stroke: string | any[], path: Path2D) {

        const canvas = document.getElementById('canvas_write') as HTMLCanvasElement
        const context = canvas.getContext('2d')!;

        const color = Class_Colors[this.current_color]

        context.imageSmoothingEnabled = false
        context.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`
        context.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`
        context.lineCap = 'square'
        context.lineJoin = 'round'

        const l = stroke.length - 1
        if (stroke.length >= 3) { // continue line
            const xc = (stroke[l].x + stroke[l - 1].x) / 2
            const yc = (stroke[l].y + stroke[l - 1].y) / 2
            context.lineWidth = 2

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

        console.log('switching to alternative', scene_code)

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

        // clear the canvas and clear the history
        this.canvas_history = [];
        this.canvas_data = Uint8ClampedArray.from({length: RAW_MASK_DIM * RAW_MASK_DIM * 4}, () => 0);
        const canvass = document.querySelectorAll('canvas');
        canvass.forEach(canvas => {
            const context = canvas.getContext('2d')!;
            context.clearRect(0, 0, canvas.width, canvas.height);
        });

    }

    set_color(c: number) {
        this.current_color = c
    }

    undoDraw() {

        this.canvas_history.pop();

        this.canvas_data = this.canvas_history[this.canvas_history.length - 1];

        const readonly_canvas = document.getElementById('canvas_readonly') as HTMLCanvasElement
        const context = readonly_canvas.getContext('2d')!;
        context.putImageData(new ImageData(this.canvas_data, RAW_MASK_DIM, RAW_MASK_DIM), 0, 0);

    }


    next_img(first_time = false) {

        if (this.loading_next) return
        this.loading_next = true;

        if (!first_time && this.current_image_ref !== null) {

            const class_image = new Array(RAW_MASK_DIM).fill(0).map(() => new Array(RAW_MASK_DIM).fill(0));

            for (let i = 0; i < RAW_MASK_DIM ** 2; i++) {

                const x = i % RAW_MASK_DIM;
                const y = Math.floor(i / RAW_MASK_DIM);

                const color = this.canvas_data.slice(i * 4, i * 4 + 4);
                if (color[3] < 255 || (color[0] === 0 && color[1] === 0 && color[2] === 0)) {
                    class_image[y][x] = Classes.Background;
                    continue;
                }

                // search for the corresponding color class
                for (const [class_nr_, color_] of Object.entries(Class_Colors)) {
                    if (color[0] === color_[0] && color[1] === color_[1] && color[2] === color_[2]) {
                        class_image[y][x] = class_nr_;
                        break;
                    }
                }

            }

            const flattened = class_image.flat();

            // send the image as a post to the backend
            const xhr = new XMLHttpRequest()
            xhr.open('POST', BASE_URL + '/update_mask/' + this.current_image_ref)
            xhr.setRequestHeader('Content-Type', 'image/png')
            xhr.send(JSON.stringify({image: flattened}))

        }

        this.clear();

        setTimeout(() => {

            // make a request to the backend to get the image
            // backend is at port 5000
            const url = BASE_URL + '/next_image'
            fetch(url)
                .then(response => response.json())
                .then(json => {

                    const scenes = json['scenes'];
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

    private async run_selection(current_position: any) {

        await this.mark_similar_pixels(current_position.x, current_position.y);

        const readonly_canvas = document.getElementById('canvas_readonly') as HTMLCanvasElement
        const context = readonly_canvas.getContext('2d')!
        context.putImageData(new ImageData(this.canvas_data, RAW_MASK_DIM, RAW_MASK_DIM), 0, 0);
        this.canvas_history.push(this.canvas_data)

    }

    private async mark_similar_pixels(x0: number, y0: number): Promise<void> {

        const tmp_canvas = document.createElement('canvas');
        tmp_canvas.width = RAW_MASK_DIM;
        tmp_canvas.height = RAW_MASK_DIM;

        const ctx = tmp_canvas.getContext('2d');

        const img = new Image();
        img.crossOrigin = 'anonymous';
        const image = document.getElementById(this.active_scene_code) as HTMLImageElement;
        img.src = image.src;

        const threshold = this.threshold;
        const visited = new Set<string>();

        // Wait for the image to load
        await new Promise<void>((resolve) => {
            img.onload = () => {
                ctx?.drawImage(img, 0, 0, RAW_MASK_DIM, RAW_MASK_DIM);
                resolve();
            };
        });

        const mask_color = Class_Colors[this.current_color]

        const colors = ctx?.getImageData(x0 - 1, y0 - 1, 3, 3).data
        if (!colors) throw new Error('color is null')

        const color = [
            (colors[0] + colors[4] + colors[8] + colors[12] + colors[16] + colors[20] + colors[24] + colors[28] + colors[32]) / 9,
            (colors[1] + colors[5] + colors[9] + colors[13] + colors[17] + colors[21] + colors[25] + colors[29] + colors[33]) / 9,
            (colors[2] + colors[6] + colors[10] + colors[14] + colors[18] + colors[22] + colors[26] + colors[30] + colors[34]) / 9
        ]

        // Copy the pixel data from the canvas into the imageData array
        const canvasData = ctx?.getImageData(0, 0, RAW_MASK_DIM, RAW_MASK_DIM).data;
        if (!canvasData) throw new Error('canvasData is null')

        const queue = new Queue();
        queue.enqueue({"x": x0, "y": y0});

        const canvas_data = this.canvas_data;

        run(function* () {

            while (queue.length) {

                const {x, y} = queue.dequeue();

                const index = (y * RAW_MASK_DIM + x) * 4;

                // MARK PIXEL
                canvas_data[index] = mask_color[0];
                canvas_data[index + 1] = mask_color[1];
                canvas_data[index + 2] = mask_color[2];
                canvas_data[index + 3] = 255;

                // yield to stop the computation after a certain amount of time
                yield;

                // Add adjacent pixels to queue
                const neighbors = [
                    {x: x - 1, y},
                    {x: x + 1, y},
                    {x, y: y - 1},
                    {x, y: y + 1}
                ];

                for (const neighbor of neighbors) {

                    const {x, y} = neighbor;
                    // Check if pixel is within bounds and hasn't been visited
                    if (x < 0 || y < 0 || x >= RAW_MASK_DIM || y >= RAW_MASK_DIM) continue;

                    const index = (y * RAW_MASK_DIM + x) * 4;
                    if (visited.has(`${x}_${y}`.toString())) continue;

                    // Check if pixel color is similar to the target color
                    const pixelColor = canvasData.slice(index, index + 3);

                    // check color2
                    if (!is_similar_color(pixelColor, color, threshold)) continue;

                    // Mark pixel as visited
                    visited.add(`${x}_${y}`.toString());

                    queue.enqueue(neighbor); // left

                }

            }


        }, 50);

    }

}
