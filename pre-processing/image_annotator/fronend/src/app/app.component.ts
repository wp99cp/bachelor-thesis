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

    ngOnInit(): void {

        this.next_img(true)

        const canvas = document.querySelectorAll('canvas')[0]
        canvas.width = 512
        canvas.height = 512

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
                // hide annotations
                const canvas_id = 'canvas';
                const canvas = document.getElementById(canvas_id)
                canvas!.style.opacity = '0';
            }

        });

        document.addEventListener('keyup', (e: KeyboardEvent) => {

            if (SCENE_CODES.includes(e.key)) this.switch_to_original();

            if (e.key === '.') {
                // hide annotations
                const canvas_id = 'canvas';
                const canvas = document.getElementById(canvas_id)
                canvas!.style.opacity = '1';
            }

            if (e.key === 'Enter') this.enter_is_down = false;
            if (e.key === '+') this.color_adding_mode = false;
            if (e.key === 'Control') this.control_is_down = false;

        });


        for (const ev of ["touchstart", "mousedown"]) {
            canvas.addEventListener(ev, (e: any) => {

                isMousedown = true
                points = [this.get_coords(e, canvas)];

            })
        }

        for (const ev of ['touchmove', 'mousemove']) {
            canvas.addEventListener(ev, async (e: any) => {

                if (!isMousedown) return
                e.preventDefault()

                if (this.color_adding_mode) {
                    await this.run_selection(this.get_coords(e, canvas));
                    return;
                }

                points.push(this.get_coords(e, canvas))

            })
        }

        for (const ev of ['touchend', 'touchleave', 'mouseup']) {

            canvas.addEventListener(ev, async (e: any) => {

                isMousedown = false

                if (this.color_adding_mode) {
                    await this.run_selection(this.get_coords(e, canvas));
                    return;
                }

                await this.draw_stroke(points)

            });
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

        const factor = RAW_MASK_DIM / 1024;
        return {"x": Math.floor(x * factor), "y": Math.floor(y * factor)}
    }


    private async draw_stroke(stroke: any[]) {

        const color = Class_Colors[this.current_color]

        const mark_pixels = (index: any) => {
            this.canvas_data[index] = color[0];
            this.canvas_data[index + 1] = color[1];
            this.canvas_data[index + 2] = color[2];
            this.canvas_data[index + 3] = color[3];
        }

        let prev_pkt = null;

        for (const p of stroke) {

            if (prev_pkt === null) {
                prev_pkt = p;
                continue;
            }

            // draw line between previous and current point
            const slope = (p.y - prev_pkt.y) / (p.x - prev_pkt.x);
            const intercept = p.y - slope * p.x;

            for (let x = Math.min(p.x, prev_pkt.x); x <= Math.max(p.x, prev_pkt.x); x++) {
                const y = slope * x + intercept;
                const index = (Math.floor(y) * RAW_MASK_DIM + Math.floor(x)) * 4;
                mark_pixels(index);
            }

            prev_pkt = p;
        }

        const context = this.get_context();
        context.putImageData(new ImageData(this.canvas_data, RAW_MASK_DIM, RAW_MASK_DIM), 0, 0);

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
        const context = this.get_context();

        // clear the canvas and clear the history
        this.canvas_data = Uint8ClampedArray.from({length: RAW_MASK_DIM * RAW_MASK_DIM * 4}, () => 0);

        context.putImageData(new ImageData(this.canvas_data, RAW_MASK_DIM, RAW_MASK_DIM), 0, 0);
        this.canvas_history = [];

    }

    private get_context() {
        const canvas = document.getElementById('canvas') as HTMLCanvasElement
        const context = canvas.getContext('2d')
        if (!context) throw new Error('context is null')
        return context;
    }

    set_color(c: number) {
        this.current_color = c
    }

    undoDraw() {

        this.canvas_history.pop();

        const context = this.get_context();

        this.canvas_data = this.canvas_history[this.canvas_history.length - 1];
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
                if (color[3] < 255) {
                    class_image[y][x] = Classes.Background;
                }

                for (const [class_code, class_color] of Object.entries(Class_Colors)) {
                    if (color[0] === class_color[0] && color[1] === class_color[1] && color[2] === class_color[2] && color[3] === class_color[3])
                        class_image[y][x] = parseInt(class_code);
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


    private async run_selection(current_position: any) {

        console.log('running selection')
        await this.mark_similar_pixels(current_position.x, current_position.y);

        const context = this.get_context();
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
