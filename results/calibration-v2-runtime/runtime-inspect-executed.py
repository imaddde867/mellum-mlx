import hashlib,importlib.metadata as md,io,json,sys,urllib.request,zipfile
from pathlib import Path
sys.path.insert(0,'scripts')
from convert import sha256
from nonexpert import write_new
root=Path('work/calibration-v2-runtime-source')
report={'status':'running','script_sha256':sha256(Path(__file__)),'distributions':{}}
for name in ['mlx','mlx-cuda-13']:
 distribution=md.distribution(name)
 if distribution.version!='0.32.3':raise ValueError('Pinned runtime version changed')
 metadata=json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{name}/0.32.3/json'))
 matches=[f for f in metadata['urls'] if 'manylinux_2_35_x86_64' in f['filename'] and ('cp312' in f['filename'] or name=='mlx-cuda-13')]
 if len(matches)!=1:raise ValueError('Ambiguous official pinned wheel')
 file=matches[0];path=root/file['filename']
 urllib.request.urlretrieve(file['url'],path)
 if sha256(path)!=file['digests']['sha256']:raise ValueError('Official wheel digest mismatch')
 wheel=zipfile.ZipFile(path); hashes={}
 for entry in wheel.namelist():
  if entry.endswith('/') or not entry.startswith('mlx/'):continue
  installed=Path(distribution.locate_file(entry))
  expected=hashlib.sha256(wheel.read(entry)).hexdigest()
  if sha256(installed)!=expected:raise ValueError('Installed runtime differs from official wheel: '+entry)
  hashes[entry]=expected
 report['distributions'][name]={'version':distribution.version,'wheel_url':file['url'],'wheel_sha256':sha256(path),'installed_files_sha256':hashes}
source=root/'mlx-0.32.3/mlx/backend/cuda/device.cpp'
text=source.read_text()
if 'env::get_var("MLX_USE_CUDA_GRAPHS", true)' not in text or 'if (!use_cuda_graphs())' not in text:raise ValueError('Native path absent in pinned source')
lib=Path(md.distribution('mlx-cuda-13').locate_file('mlx/lib/libmlx.so'))
if b'MLX_USE_CUDA_GRAPHS' not in lib.read_bytes():raise ValueError('Installed binary lacks native switch')
report.update(status='complete',source_url='https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/cuda/device.cpp',
 source_sha256=sha256(source),source_archive_sha256=sha256(root/'mlx-v0.32.3.tar.gz'),
 installed_official_wheels_identical=True,native_switch_in_binary=True)
write_new(Path('results/calibration-v2-runtime/runtime-inspection.json'),report)
