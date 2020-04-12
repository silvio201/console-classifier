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

export_file_url = 'https://drive.google.com/u/0/uc?id=1r9L5ojNKHGOq2w8kVX9FSj2ffue-zA8y&export=download'
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

@app.route("/analyze", methods=["POST"])
async def analyze(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)



def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    x,y,losses = learn.predict(img)
    classes = learn.data.classes
    results = list(map(round, (map(float, losses*100))))
    responsestring = """<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
            <title>Console Classifier</title>
        </head>
        <body>
        <div>
            <h1 class="display-1">Results</h1>"""
    for i in range(10):
        responsestring = responsestring + """<div class="progress">
                    <div class="progress-bar" role="progressbar" style="width:"""
        responsestring = responsestring + str(results[i]) + """%" aria-valuenow=" """+ str(results[i]) + """ "  aria-valuemin="0" aria-valuemax="100">
                        """
        responsestring = responsestring + str(classes[i])
        responsestring = responsestring + """
                        </div>
                    </div>"""
    responsestring = responsestring + """</div>   
            <form action="/">
                <input type="submit" value="Analyze another picture">
            </form>        
        </body>
        </html>
        """
    
    return HTMLResponse(
        responsestring
    )
    # return JSONResponse({
    #     "predictions": sorted(
    #         zip(learn.data.classes, map(float, losses)),
    #         key=lambda p: p[0]
    #     ),
    #     "results": [(label, prob) for label, prob in zip(learn.data.classes, map(round, (map(float, losses*100))))]   
    # })

    


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
            <title>Console Classifier</title>
        </head>
        <body>
            <h1 class="display-1">Console Classifier by Felix Anzengruber</h1>
            <div>
                <form action="/analyze" method="post" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="fileupload">Select image to upload:</label>
                        <input class="form-control" id="fileupload" type="file" name="file"><br>
                        <input class="btn btn-primary" type="submit" value="Upload and Analyze Image">
                    </div>
                </form>
            </div>
        </body>
        </html>
    """)


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
