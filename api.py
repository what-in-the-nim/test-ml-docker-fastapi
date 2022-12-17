import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from ml import load_model, extract_image, get_preprocessor, most_probability_class

model = load_model()
preprocessor = get_preprocessor()

description = 'MLAPI helps you to inference input to machine learning model. 🚀🚀🚀'


app = FastAPI(
    title='MLAPI',
    description=description
)

@app.get('/welcome', summary='Nice warm welcome to MLAPI')
async def welcome():
    return JSONResponse('Welcome to MLAPI')


@app.post('/predict/text/', summary='Inference text input')
async def inference_text(text: str):
    return_text = text.swapcase() #Some text model manipulation
    return JSONResponse(return_text)

@app.post('/predict/image/', summary='Inference image input')
async def inference_image(package: UploadFile = File()):
    contents = await package.read()
    image = extract_image(contents)
    image = preprocessor(image)
    image = torch.unsqueeze(image, 0)
    result = model(image).detach().numpy()
    cls = most_probability_class(result)

    return JSONResponse(cls)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('api:app', host='0.0.0.0', port=1234, reload=True)