from minio import Minio
import pathlib
from minio.error import S3Error
import subprocess
import tempfile
import string
import random
from pony import orm
import argparse
import os
import json
import train_dreambooth
from natsort import natsorted
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import gradio as gr
import sys
import glob
import datetime

MAX_TRAIN_STEPS = 2000

class DreamboothPathManager():
  def __init__(self, rootDir, personId):
    self.rootDir = rootDir
    self.personId = personId

  def getBaseDir(self):
    return f"{self.rootDir}"

  def getModelDir(self):
    return f"{self.getBaseDir()}/content"

  # FIX-ME this is reall the weights output directory where each subdir named K is the model trained at K steps.
  def getOutputDir(self):
    return f"{self.getBaseDir()}/content/stable_diffusion_weights"

  def getGeneratedImages(self):
    return f"{self.getBaseDir()}/generated-images"

  # FIX-ME - the weights dir is relative to the MAX_TRAIN_STEPS (for k steps but in directory called K)
  def getWeightsDir(self):
    #    return natsorted(glob.glob(self.getOutputDir() + os.sep + "*"))[-1]
    return f'{self.getOutputDir()}/{MAX_TRAIN_STEPS}'

  def getSourceImagesDir(self):
    return f"/{self.getBaseDir()}/source_images"

  def getClassDataDir(self):
    return f"/{self.getBaseDir()}/class_data"


class DreamboothModelMaker():
  def __init__(self, rootDir, personId):

    self.pathManager = DreamboothPathManager(rootDir, personId)
    self.MODEL_NAME = "runwayml/stable-diffusion-v1-5"
    self.personId = personId
    self.tokenId = 'sks'
    self.OUTPUT_DIR = self.pathManager.getOutputDir()

    os.makedirs(self.OUTPUT_DIR, exist_ok=True)
    print(f"[*] Weights will be saved at {self.pathManager.getWeightsDir()}")

  def getConceptsList(self):
    # FIX-ME for non-person
    conceptsList = [
        {
            "instance_prompt":      f"photo of {self.tokenId} person",
            "class_prompt":         "photo of a person",
            "instance_data_dir":    f"{self.pathManager.getSourceImagesDir()}",
            "class_data_dir":       f"{self.pathManager.getClassDataDir()}"
        },
    ]

    return conceptsList

  def getConceptsListFilePath(self):
    return f"{self.pathManager.getModelDir()}/concepts_list.json"

  def setupInstanceData(self):
    # `class_data_dir` contains regularization images
    for concept in self.getConceptsList():
      os.makedirs(concept["instance_data_dir"], exist_ok=True)

    with open(self.getConceptsListFilePath(), "w") as f:
      json.dump(self.getConceptsList(), f, indent=4)

  def trainModel(self):
    args = [
        "--pretrained_model_name_or_path", self.MODEL_NAME,
        "--pretrained_vae_name_or_path", "stabilityai/sd-vae-ft-mse",
        "--output_dir", self.OUTPUT_DIR,
        "--revision=fp16",
        "--with_prior_preservation",
        "--prior_loss_weight=1.0",
        "--seed=1337",
        "--resolution=512",
        "--train_batch_size=1",
        "--train_text_encoder",
        "--mixed_precision=fp16",
        "--use_8bit_adam",
        "--gradient_accumulation_steps=1",
        "--gradient_checkpointing",
        "--learning_rate=1e-6",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps=0",
        "--num_class_images=50",
        "--sample_batch_size=4",
        f"--max_train_steps={MAX_TRAIN_STEPS}",  
        "--save_interval=10000",
        '--concepts_list', self.getConceptsListFilePath()
    ]
    train_dreambooth.main(train_dreambooth.parse_args(args))


class DreamboothImageGenerator:
  def __init__(self, rootDir, personId):
    self.pathManager = DreamboothPathManager(rootDir, personId)

    self.personId = personId

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                              beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    self.pipe = StableDiffusionPipeline.from_pretrained(self.pathManager.getWeightsDir(
    ), scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16).to("cuda")
    self.g_cuda = torch.Generator(device='cuda')
    seed = 52362  # @param {type:"number"}
    self.g_cuda.manual_seed(seed)
    self.tokenId = 'sks'

  def generateImages(self, prompt = "photo of @me in a bucket", subdir = "x"):
    #@title Run for generating images.
    # @param {type:"string"}
    seed = 52362  # @param {type:"number"}
    self.g_cuda.manual_seed(seed)

    prompt = self.replacePromptPlaceholder(prompt, suffix = "face like @me hair like @me eyes like @me")
    negative_prompt = ""  # @param {type:"string"}
    num_samples = 4  # @param {type:"number"}
    guidance_scale = 7.5  # @param {type:"number"}
    num_inference_steps = 50  # @param {type:"number"}
    height = 512  # @param {type:"number"}
    width = 512  # @param {type:"number"}

    print(f"{prompt}")
    with autocast("cuda"), torch.inference_mode():
        images = self.pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=self.g_cuda
        ).images

    # Place images in subdir and put the prompt in prompt.txt
    directory = f"{self.pathManager.getGeneratedImages()}/{subdir}"
    os.makedirs(directory, exist_ok=True)
    imagePrefix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))

    for i, image in enumerate(images):
      image.save(f"{directory}/{imagePrefix}-img{i}.png")
    with open(f"{directory}/{imagePrefix}-prompt.txt", "w") as text_file:
      print(prompt, file = text_file)
 
  def replacePromptPlaceholder(self, prompt, placeholder = "@me", suffix = ""):
    prompt = f"{suffix} {prompt}"
    return prompt.replace(placeholder, f'{self.tokenId} person')

  def runWebApp(self):
    def inference(prompt, negative_prompt, num_samples, height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
        # Start at same seed each time so we get consistent results
        seed = 52362  # @param {type:"number"}
        self.g_cuda.manual_seed(seed)
        prompt = self.replacePromptPlaceholder(prompt)
        print(f"prompt: {prompt}")
        negative_prompt = self.replacePromptPlaceholder(negative_prompt)
        with torch.autocast("cuda"), torch.inference_mode():
            return self.pipe(
                prompt, height=int(height), width=int(width),
                negative_prompt=negative_prompt,
                num_images_per_prompt=int(num_samples),
                num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                generator=self.g_cuda
            ).images

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt", value=f"photo of @me in a bucket")
                negative_prompt = gr.Textbox(label="Negative Prompt", value="")
                run = gr.Button(value="Generate")
                with gr.Row():
                    num_samples = gr.Number(label="Number of Samples", value=4)
                    guidance_scale = gr.Number(
                        label="Guidance Scale", value=7.5)
                with gr.Row():
                    height = gr.Number(label="Height", value=512)
                    width = gr.Number(label="Width", value=512)
                num_inference_steps = gr.Slider(label="Steps", value=50)
            with gr.Column():
                gallery = gr.Gallery()

        run.click(inference, inputs=[prompt, negative_prompt, num_samples,
                  height, width, num_inference_steps, guidance_scale], outputs=gallery, api_name = "api")
        demo.launch(debug=True)




db = orm.Database()


class User(db.Entity):
  username = orm.Required(str, unique=True)
  dreamBoothContainers = orm.Set(lambda: DreamboothContainer)

  def create(username):
    return User(username=username)

  def newDreamboothContainer(self):
    return DreamboothContainer(user=self, data_location="db")


class DreamboothContainer(db.Entity):
  user = orm.Required(User)
  data_location = orm.Required(str)
  dreambooths = orm.Set(lambda: Dreambooth)

  def newDreambooth(self, realName):
    tokenName = ''.join(random.choices(
        string.ascii_lowercase + string.digits, k=5))
    return Dreambooth(container=self, real_name=realName, token_name=tokenName, processing_state=Dreambooth.STATE_WAITING_FOR_IMAGES)


class Dreambooth(db.Entity):
  container = orm.Required(DreamboothContainer)
  real_name = orm.Required(str)
  token_name = orm.Required(str, unique=True)
  processing_state = orm.Required(int)
  queries = orm.Set(lambda: Query)

  STATE_WAITING_FOR_IMAGES = 0
  STATE_WAITING_FOR_MODEL = 1
  STATE_BUILDING_MODEL = 2
  STATE_READY_FOR_QUERIES = 3

  def addQuery(self, prompt):
    path = ''.join(random.choices(
        string.ascii_lowercase + string.digits, k=10))
    return Query(dreambooth=self, prompt=prompt, path=path)


class Query(db.Entity):
  dreambooth = orm.Required(Dreambooth)
  prompt = orm.Required(str)
  path = orm.Required(str)


db.bind(provider='mysql', host='127.0.0.1', user='root',
        passwd='kibbtzi', db='dreambooth', port=3306)
db.generate_mapping(create_tables=True)


class DreamboothRemoteRepository:
  def __init__(self, username, containerDict, boothDict):
    bucket = f"qqq-{containerDict['data_location']}"
    self.userDataPath = f'minio:{bucket}/{username}/{boothDict["token_name"]}'

  def downloadBooth(self, destinationDirectory):
    print(f'downloading {self.userDataPath} to {destinationDirectory}')
    completedProcess = subprocess.run(
        ['rclone', 'copy', '--progress', self.userDataPath, destinationDirectory], capture_output=True, check=True)
    completedProcess.check_returncode()

  def uploadBooth(self, sourceDirectory):
    print('uploading booth')
    completedProcess = subprocess.run(
        ['rclone', 'copy', '--progress', sourceDirectory, self.userDataPath], capture_output=True, check=True)
    completedProcess.check_returncode()


class Driver:
  def createUser(username):
    with orm.db_session:
      u1 = User.create(username)
      dbc = u1.newDreamboothContainer()
      db = dbc.newDreambooth("default")

  def newBooth(username, realName):
    with orm.db_session:
      user = User.get(username=username)
      dbc = user.dreamBoothContainers.select().first()
      db = dbc.newDreambooth(realName)
      print(db)

  def listBooths(username):
    with orm.db_session:
      user = User.get(username=username)
      for container in user.dreamBoothContainers:
        for dreambooth in container.dreambooths:
          print(f"{dreambooth} {dreambooth.real_name} {dreambooth.token_name}")

  def getBooth(username, id):
    print(f"getBooth: {username}, {id}")
    with orm.db_session:
      query = orm.select(
          (dreamBoothContainer, dreambooth)
          for user in User for dreamBoothContainer in user.dreamBoothContainers for dreambooth in dreamBoothContainer.dreambooths
          if user.username == username and dreambooth.id == id
      )
      return (query.first()[0].to_dict(), query.first()[1].to_dict())

  def buildModel(username, boothId):
    (container, dreambooth) = Driver.getBooth(username, boothId)
    print(dreambooth)

    with tempfile.TemporaryDirectory() as tempdir:
      remoteRepository = DreamboothRemoteRepository(username, container, dreambooth)
      remoteRepository.downloadBooth(tempdir)

      modelMaker = DreamboothModelMaker(
          rootDir=tempdir, personId=dreambooth['token_name'])
      modelMaker.setupInstanceData()
      modelMaker.trainModel()

      remoteRepository.uploadBooth(tempdir)

  def startBoothWebapp(username, boothId):
    (container, dreambooth) = Driver.getBooth(username, boothId)

    # Pull down model
    remoteRepository = DreamboothRemoteRepository(username, container, dreambooth)
    with tempfile.TemporaryDirectory() as tempdir:
      remoteRepository.downloadBooth(tempdir)
      generator = DreamboothImageGenerator(
          rootDir=tempdir, personId=dreambooth['token_name'])
      generator.runWebApp()
      # starting building model

  def generateImages(username, boothId, prompts):
    (container, dreambooth) = Driver.getBooth(username, boothId)

    # Pull down model
    remoteRepository = DreamboothRemoteRepository(username, container, dreambooth)
    with tempfile.TemporaryDirectory() as tempdir:
      remoteRepository.downloadBooth(tempdir)
      generator = DreamboothImageGenerator(
          rootDir=tempdir, personId=dreambooth['token_name'])
      imagesDir = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S");
      for prompt in prompts:
        generator.generateImages(prompt = prompt, subdir = imagesDir)
      remoteRepository.uploadBooth(tempdir)
      # starting building model

  def processDreamBooths():
    # for all dreambooths is state STATE_WAITING_FOR_MODEL

      # set to STATE_BUILDING_MODEL

      # build the model

      # set to STATE_READY_FOR_QUERIES
    return

  # FIX-ME only allow jpg and png
  def uploadPhotos(username, boothId, files):
    (container, dreambooth) = Driver.getBooth(username, boothId)

    client = Minio(
        "192.168.1.138:9000",
        access_key="XIy1fs98fvDuItAr",
        secret_key="cBFGUVfPfG0oZdOl4gEFB8rTJPsGaJj6",
        secure=False
    )

    imagesBucket = f"qqq-{container['data_location']}"

    if not client.bucket_exists(imagesBucket):
      client.make_bucket(imagesBucket)

    pathManager = DreamboothPathManager(username, dreambooth['token_name'])
    # Upload each of the files
    for file in files:
      filePath = pathlib.PurePath(file)
      objectName = f"{pathManager.getSourceImagesDir()}/{filePath.name}"
      print(f'uploading {file} to {objectName}')

      client.fput_object(imagesBucket, objectName, filePath)

  def createPromptedDerivatives():
    return


def executeListComand(args):
  Driver.listBooths(args.username)


def executeShowBoothCommand(args):
  (container, dreambooth) = Driver.getBooth(args.username, args.booth_id)
  print(f"container: {container} booth: {dreambooth}")


def executeBuildModel(args):
  (container, dreambooth) = Driver.getBooth(args.username, args.booth_id)
  print(f"container: {container} booth: {dreambooth}")
  Driver.buildModel(args.username, args.booth_id)


def executeStartBoothWebapp(args):
  (container, dreambooth) = Driver.getBooth(args.username, args.booth_id)
  print(f"container: {container} booth: {dreambooth}")
  Driver.startBoothWebapp(username=args.username, boothId=args.booth_id)


def executeNewBooth(args):
  Driver.newBooth(username=args.username, realName=args.real_name)


def executeUploadFiles(args):
  files = glob.glob(os.path.expanduser(args.file_path_glob))
  Driver.uploadPhotos(args.username, args.booth_id, files)

def loadFileAsList(filename):
  with open(filename) as f:
    list = [line.rstrip() for line in f]
    return list

def executeGenerateImages(args):
  (container, dreambooth) = Driver.getBooth(args.username, args.booth_id)
  print(f"container: {container} booth: {dreambooth}")
  # add in prompts
  Driver.generateImages(boothId = args.booth_id, username = args.username, prompts = loadFileAsList(args.prompt_file))

def parseArgs(args):
  # create the top-level parser
  parser = argparse.ArgumentParser(
      description="Simple example of a training script.")
  subparsers = parser.add_subparsers()

  # create the parser for the "list" command
  parserList = subparsers.add_parser('list-booths')
  parserList.add_argument('--username', type=str, required=True)
  parserList.set_defaults(func=executeListComand)

  parserList = subparsers.add_parser('show-booth')
  parserList.add_argument('--username', type=str, required=True)
  parserList.add_argument('--booth-id', type=int, required=True)
  parserList.set_defaults(func=executeShowBoothCommand)

  parserList = subparsers.add_parser('upload-files')
  parserList.add_argument('--username', type=str, required=True)
  parserList.add_argument('--booth-id', type=int, required=True)
  parserList.add_argument('--file-path-glob', type=str, required=True)
  parserList.set_defaults(func=executeUploadFiles)

  parserList = subparsers.add_parser('build-model')
  parserList.add_argument('--username', type=str, required=True)
  parserList.add_argument('--booth-id', type=int, required=True)
  parserList.set_defaults(func=executeBuildModel)

  parserList = subparsers.add_parser('start-web-app')
  parserList.add_argument('--username', type=str, required=True)
  parserList.add_argument('--booth-id', type=int, required=True)
  parserList.set_defaults(func=executeStartBoothWebapp)

  parserList = subparsers.add_parser('new-booth')
  parserList.add_argument('--username', type=str, required=True)
  parserList.add_argument('--real-name', type=str, required=True)
  parserList.set_defaults(func=executeNewBooth)

  parserList = subparsers.add_parser('generate-images')
  parserList.add_argument('--username', type=str, required=True)
  parserList.add_argument('--booth-id', type=int, required=True)
  parserList.add_argument('--prompt-file', type=str, required=True, help = 'path to file containing prompts')
  parserList.set_defaults(func=executeGenerateImages)

  subCommandArgs = parser.parse_args(args)
  subCommandArgs.func(subCommandArgs)


parseArgs(sys.argv[1:])

'''

#orm.set_sql_debug(True)

need to set up rclone...

mysql -h 127.0.0.1 -u root -p

boothstates

WAITING_FOR_PHOTOS, GENERATING_MODEL, READY_FOR_INFERENCE


MESSAGE_QUEUE




  sudo docker run  -d  -p 9000:9000    -p 9090:9090    --name minio    -v ~/minio/data:/data    -e "MINIO_ROOT_USER=ROOTNAME"    -e "MINIO_ROOT_PASSWORD=CHANGEME123"    quay.io/minio/minio server /data --console-address ":9090"



'''
