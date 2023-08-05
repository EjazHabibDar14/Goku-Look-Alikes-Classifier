from fastai.vision.all import *
import gradio as gr

def is_goku(x):
  return x[0].issuper()

learn =load_learner('model.pkl')

categories=('Bardock', 'Goku', 'Goku Black', 'Goku Jr.', 'Goten', 'Turles')

def classify_image(img):
  pred,idx,probs = learn.predict(img)
  return dict(zip(categories,map(float,probs)))

image=gr.inputs.Image(shape=(192,192))
label=gr.outputs.Label()
examples =['Goku.jpg', 'Goku Black.jpg', 'Turles.jpg', 'Bardock.jpg', 'Goku Jr..jpg', 'Goten.jpg']
title = "Goku Lookalikes Classifier"

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples, title= title)
intf.launch(inline=False)