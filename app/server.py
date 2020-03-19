import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.routing import Route

export_file_url = 'https://drive.google.com/file/d/1r9L5ojNKHGOq2w8kVX9FSj2ffue-zA8y/view?usp=sharing'
export_file_name = 'export.pkl'

classes = ['NintendoSwitch', 'NintendoWiiU', 'NintendoWii', 'NintendoGamecube', 'Xbox360', 'XboxOne', 'Playstation1', 'Playstation2', 'Playstation3', 'Playstation4']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)

async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

asyncio.set_event_loop(asyncio.new_event_loop())
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
#loop.close()

#learn = setup_learner()

@app.route("/analyze", methods=["POST"])
async def analyze(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


# @app.route("/classify-url", methods=["GET"])
# async def classify_url(request):
#     bytes = await get_bytes(request.query_params["url"])
#     context = {
#         "request": request, 
#         "data": predict_image_from_bytes(bytes)
#     }
#     return templates.TemplateResponse('show_predictions.html', context=context)


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    x,y,losses = learn.predict(img)   
    return JSONResponse({
        "predictions": sorted(
            zip(learn.data.classes, map(float, losses)),
            key=lambda p: p[0]
        ),
        "results": [(label, prob) for label, prob in zip(learn.data.classes, map(round, (map(float, losses*100))))]   
    })

# @app.route('/')
# async def homepage(request):
#     html_file = path / 'templates' / 'index.html'
#     return templates.TemplateResponse("index.html", {"request": request})

@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Console Classifier</title>
        </head>
        <body>
            <div>
                <form action="/analyze" method="post" enctype="multipart/form-data">
                    <div>
                        Select image to upload:
                        <input type="file" name="file">
                        <input type="submit" value="Upload and Analyze Image">
                    </div>
                </form>

            </div>
        </body>
        </html>
    """)


# @app.route("/form")
# def redirect_to_homepage(request):
#     return RedirectResponse("/")

# routes = [
#     Route("/", endpoint=homepage),
#     Route("/upload", endpoint=upload, methods=["POST"]),
#     Route("/classify-url", endpoint=classify_url, methods=["GET"]),
#     Route("/form", endpoint=redirect_to_homepage),
# ]



# if __name__ == '__main__':
#     if 'serve' in sys.argv:
#         uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")