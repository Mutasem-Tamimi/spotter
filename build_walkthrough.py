"""Create an explicitly synthetic-narrated code walkthrough for Loom upload.

Requires Windows System.Speech, Pillow (already installed), and ffmpeg on PATH.
The generated MP4 is a project explanation, not a recording of the applicant.
"""
from pathlib import Path
import json
import subprocess
import textwrap
import wave

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "tmp/walkthrough"
WORK.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "output/video"
OUT.mkdir(parents=True, exist_ok=True)
FONT = Path("C:/Windows/Fonts")
NAVY = "#173044"
TEAL = "#087E8B"
WHITE = "#FFFFFF"
MUTED = "#B9D0DD"


def font(size, bold=False, mono=False):
    return ImageFont.truetype(str(FONT / ("consola.ttf" if mono else "arialbd.ttf" if bold else "arial.ttf")), size)


def block(draw, xy, text, size=33, color=WHITE, width=83, spacing=14):
    x, y = xy
    for paragraph in text.split("\n"):
        for line in textwrap.wrap(paragraph, width=width) or [""]:
            draw.text((x, y), line, font=font(size), fill=color)
            y += size + spacing
    return y


def code_excerpt(draw, lines, y=370):
    draw.rounded_rectangle((76, y - 25, 1844, 923), radius=18, fill="#0B1D2A")
    for line in lines:
        draw.text((106, y), line, font=font(27, mono=True), fill="#CBECE8")
        y += 43


def source_lines(name, begin, end):
    lines = (ROOT / name).read_text().splitlines()
    first = next(i for i, line in enumerate(lines) if begin in line)
    last = next(i for i in range(first + 1, len(lines)) if end in lines[i])
    return [f"{i+1:3}  {lines[i]}" for i in range(first, last + 1)]


def make_slides():
    source = (ROOT / "walkthrough_script.txt").read_text()
    sections = []
    for i in range(1, 7):
        part = source.split(f"\n{i}. ", 1)[1]
        if i < 6:
            part = part.split(f"\n{i+1}. ", 1)[0]
        title, narration = part.strip().split("\n", 1)
        sections.append({"title": title, "narration": " ".join(narration.split())})
    for i, section in enumerate(sections):
        img = Image.new("RGB", (1920, 1080), NAVY)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 1920, 12), fill=TEAL)
        draw.text((76, 56), "SPOTTER / FREIGHT RATE PREDICTION", font=font(25, bold=True), fill=MUTED)
        draw.text((76, 122), section["title"], font=font(67, bold=True), fill=WHITE)
        draw.text((76, 1008), "Computer-generated narration | Actual project code and saved results", font=font(23), fill=MUTED)
        draw.text((1770, 1008), f"{i+1} / 6", font=font(25), fill=MUTED)
        if i == 0:
            for x, value, label in [(76,"48,000","labeled loads / Jan-Oct"),(700,"12,000","final predictions / Nov-Dec"),(1325,"31","fixed December loads")]:
                draw.rounded_rectangle((x,260,x+520,510),radius=18,fill="#244558")
                draw.text((x+28,292),value,font=font(76,True),fill="#6BE1D4")
                draw.text((x+28,413),label,font=font(29),fill=WHITE)
            block(draw,(80,575),"Target: total posted price in US dollars\nKey finding: distance and equipment explain much of the fitted model's behavior.\nReport both typical error and large-error sensitivity.",size=39,width=78,spacing=20)
        elif i == 1:
            block(draw,(80,240),"292 negative weights | 300 missing weights in labeled data\nCorrection is an assumption supported by relationship checks, not proof.",size=34,width=92)
            lines=source_lines('freight_model.py','weight = pd.to_numeric','negative = weight.lt(0)')
            code_excerpt(draw,lines,y=430)
            block(draw,(110,720),"Missing values stay NaN for CatBoost.\nSource CSVs and valid negative longitudes are preserved.",size=34,width=86)
        elif i == 2:
            block(draw,(80,240),"Train Jan-Jun: 28,806  |  Dev Jul-Aug: 9,671  |  Test Sep-Oct: 9,523\nSame rows for all models; seed 42 controls training randomness.",size=32,width=100)
            lines=source_lines('freight_model.py','splits = {','if any(part.empty')[:-1]
            code_excerpt(draw,lines,y=430)
            block(draw,(110,738),"Preprocessing is learned only from fitting rows.\nThe dev set selects settings; test data is not used for early stopping.",size=32,width=94)
        elif i == 3:
            import csv
            with (ROOT/'outputs/comparison/comparison_metrics.csv').open() as f:
                metrics=list(csv.DictReader(f))
            draw.text((90,267),"Model",font=font(32,True),fill=MUTED)
            draw.text((680,267),"Dev MAE",font=font(32,True),fill=MUTED)
            draw.text((1210,267),"Test MAE",font=font(32,True),fill=MUTED)
            for j,model in enumerate(['Baseline','CatBoost','XGBoost','FFN']):
                d=next(r for r in metrics if r['model']==model and r['phase']=='dev')
                t=next(r for r in metrics if r['model']==model and r['phase']=='test')
                y=338+j*104
                draw.text((90,y),model,font=font(44,model=='CatBoost'),fill=WHITE)
                draw.text((680,y),f"${float(d['MAE']):,.2f}",font=font(44),fill=WHITE)
                draw.text((1210,y),f"${float(t['MAE']):,.2f}",font=font(44),fill="#6BE1D4")
            block(draw,(90,802),"Submitted: CatBoost depth 6, 94 trees. Native categories and nulls.\nXGBoost wins dev MAE. Reused test results are not fresh validation.",size=31,width=100)
        elif i == 4:
            block(draw,(80,235),"freight_model.py: reusable feature preparation\npredict_validation.py: final fit and ID-aligned predictions",size=32,width=100)
            lines=source_lines('predict_validation.py','pipeline.fit(labeled','validate_predictions(filled)')
            code_excerpt(draw,lines,y=405)
            block(draw,(110,787),"complete_submission.py: predict December and run score.py\nNotebook 05: execute the workflow and build the report",size=31,width=95)
        else:
            chart=Image.open(ROOT/'scorer_results/candidate_december.png').convert('RGB')
            chart.thumbnail((1630,605))
            img.paste(chart,((1920-chart.width)//2,238))
            block(draw,(82,864),"$818.09-$826.55 | Same load, only date changes\n12,000 + 31 rows validated. Hidden-target accuracy remains unknown.",size=31,width=100)
        img.save(WORK/f'slide-{i+1}.png')
        (WORK/f'narration-{i+1}.txt').write_text(section['narration'],encoding='utf-8')
    (WORK/'sections.json').write_text(json.dumps(sections,indent=2))
    return sections


def main():
    sections=make_slides()
    # Standard installed English voice; no impersonation or voice cloning.
    powershell = r'''
param([string]$Work)
Add-Type -AssemblyName System.Speech
$narrator = New-Object System.Speech.Synthesis.SpeechSynthesizer
$english = $narrator.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'en-*' } | Select-Object -First 1
if (-not $english) { throw 'An installed English voice is required.' }
$narrator.SelectVoice($english.VoiceInfo.Name)
$narrator.Rate = 2
foreach ($i in 1..6) {
    $audioPath = Join-Path $Work "narration-$i.wav"
    $textPath = Join-Path $Work "narration-$i.txt"
    $narrator.SetOutputToWaveFile($audioPath)
    $narrator.Speak([System.IO.File]::ReadAllText($textPath))
    $narrator.SetOutputToNull()
}
$narrator.Dispose()
'''
    ps=WORK/'narrate.ps1';ps.write_text(powershell)
    subprocess.run(['powershell','-NoProfile','-File',str(ps),'-Work',str(WORK)],check=True)
    clips=[]; timeline=[];elapsed=0
    for i, section in enumerate(sections,1):
        wav=WORK/f'narration-{i}.wav'
        with wave.open(str(wav)) as stream:
            duration=stream.getnframes()/stream.getframerate()+.6
        clip=WORK/f'clip-{i}.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-loop','1','-framerate','10',
                        '-i',str(WORK/f'slide-{i}.png'),'-i',str(wav),'-t',str(duration),
                        '-c:v','libx264','-preset','veryfast','-tune','stillimage','-crf','23',
                        '-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-af','apad',str(clip)],check=True)
        clips.append(f"file '{clip.as_posix()}'")
        timeline.append({'title':section['title'],'start_seconds':round(elapsed,2),'duration_seconds':round(duration,2)})
        elapsed+=duration
        print(f'Clip {i}: {duration:.1f} seconds',flush=True)
    concat=WORK/'clips.txt';concat.write_text('\n'.join(clips))
    video=OUT/'spotter_walkthrough.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(concat),
                    '-c','copy','-movflags','+faststart',str(video)],check=True)
    if not 120 <= elapsed <= 180:
        raise ValueError(f'Walkthrough must last 2-3 minutes; got {elapsed:.1f} seconds')
    (OUT/'walkthrough_timeline.json').write_text(json.dumps({'duration_seconds':round(elapsed,2),
        'narration':'Standard Windows English synthetic voice; not the applicant\'s voice',
        'segments':timeline},indent=2))
    print(f'Saved {video} ({elapsed:.1f} seconds)',flush=True)


if __name__=='__main__':
    main()
