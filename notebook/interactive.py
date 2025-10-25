from ipycanvas import Canvas
from ipywidgets import VBox, Button

# Create canvas
canvas = Canvas(width=280, height=280)
canvas.layout.width = canvas.layout.height = '200px'
canvas.fill_style = 'black'
canvas.fill_rect(0, 0, 280, 280)
canvas.stroke_style = 'white'
canvas.line_width = 10

# Drawing state
drawing = False

def on_mouse_down(x, y):
    global drawing
    drawing = True
    canvas.begin_path()
    canvas.move_to(x, y)

def on_mouse_move(x, y):
    if drawing:
        canvas.line_to(x, y)
        canvas.stroke()

def on_mouse_up(x, y):
    global drawing
    drawing = False

canvas.on_mouse_down(on_mouse_down)
canvas.on_mouse_move(on_mouse_move)
canvas.on_mouse_up(on_mouse_up)

# Clear button
clear_button = Button(description='Clear')

def clear_canvas(b):
    global drawing
    drawing = False
    canvas.clear()
    canvas.fill_style = 'black'
    canvas.fill_rect(0, 0, 280, 280)
    canvas.stroke_style = 'white'
    canvas.line_width = 10

clear_button.on_click(clear_canvas)

VBox([canvas, clear_button])