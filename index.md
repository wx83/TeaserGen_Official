__TeaserGen: Generating Teasers for Long Documentaries__
{:.center .larger}

[Weihan Xu](https://wx83.github.io/) <sup>1</sup> &emsp;
[Paul Pu Liang](https://pliang279.github.io/) <sup>2</sup> &emsp;
[Haven Kim](https://havenpersona.github.io/) <sup>3</sup> &emsp;\\
[Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/) <sup>3</sup> &emsp;
[Taylor Berg-Kirkpatrick](https://cseweb.ucsd.edu/~tberg/) <sup>3</sup> &emsp;
[Hao-Wen Dong](https://salu133445.github.io/) <sup>4</sup>  
<sup>1</sup> Duke University &nbsp;&nbsp;&nbsp;<sup>2</sup> MIT &nbsp;&nbsp;&nbsp;<sup>3</sup> University of California San Diego &nbsp;&nbsp;&nbsp;  
<sup>4</sup> University of Michigan

[Codebase](https://github.com/wx83/TeaserGen_Official/tree/main)

{:.center}





## Contents

1) [Section 1: Qualitative Examples](#section-1-qualitative-examples)

2) [Section 2: Zero-shot Examples](#section-2-zero-shot-examples)

3) [Section 3: Dataset Examples](#section-3-dataset-examples)

4) [Section 4: More Examples](#section-4-more-examples)

## Section 1: Qualitative Examples

**Descriptions**: In this section, we present some qualitative examples generated with TeaserGen-PT and TeaserGen-LR. 


### Example 1
<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>Input Video</strong></td>
    <td style="text-align: center;"><strong>TeaserGen-PT</strong></td>
    <td style="text-align: center;"><strong>TeaserGen-LR</strong></td>
    <td style="text-align: center;"><strong>Ground Truth</strong></td>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/OhM-Aeyqp6c?start=98" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/OhM-Aeyqp6c/gpt_univtg_title_threshold/gpt_univtg_title_threshold_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/OhM-Aeyqp6c/gpt_transformer_beamsearch_dp/gpt_transformer_beamsearch_dp_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/OhM-Aeyqp6c/GT/intro.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

<div style="text-align: center; margin: 0; padding: 0;">
  <img src="https://54321anonymous.github.io/ICLR2025/oh_all.png" style="max-width: 100%; height: auto; margin: 0 auto; display: block;">
</div>

---

### Example 2

<div style="overflow-x: auto;">
  <table>
    <tr>
      <td style="text-align: center;"><strong>Input Video</strong></td>
      <td style="text-align: center;"><strong>TeaserGen-PT</strong></td>
      <td style="text-align: center;"><strong>TeaserGen-LR</strong></td>
      <td style="text-align: center;"><strong>Ground Truth</strong></td>
    </tr>
    <tr>
      <td>
        <iframe width="320" height="240" src="https://www.youtube.com/embed/UzivaxYf1iM?start=48" 
        frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
      </td>
      <td>
        <video width="320" height="240" controls>
          <source src="https://54321anonymous.github.io/ICLR2025/versionA/UzivaxYf1iM/gpt_univtg_title_threshold/gpt_univtg_title_threshold_demo.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
      </td>
      <td>
        <video width="320" height="240" controls>
          <source src="https://54321anonymous.github.io/ICLR2025/versionA/UzivaxYf1iM/gpt_transformer_beamsearch_dp/gpt_transformer_beamsearch_dp_demo.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
      </td>
      <td>
        <video width="320" height="240" controls>
          <source src="https://54321anonymous.github.io/ICLR2025/versionA/UzivaxYf1iM/GT/intro.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
      </td>
    </tr>
  </table>
</div>

<div style="text-align: center; margin: 0; padding: 0;">
  <img src="https://54321anonymous.github.io/ICLR2025/uzi_all.png" style="max-width: 100%; height: auto; margin: 0 auto; display: block;">
</div>

---

### Example 3
<div style="overflow-x: auto;">
  <table>
    <tr>
      <td style="text-align: center;"><strong>Input Video</strong></td>
      <td style="text-align: center;"><strong>TeaserGen-PT</strong></td>
      <td style="text-align: center;"><strong>TeaserGen-LR</strong></td>
      <td style="text-align: center;"><strong>Ground Truth</strong></td>
    </tr>
    <tr>
      <td>
        <iframe width="320" height="240" src="https://www.youtube.com/embed/89xTTczbv0E?start=120" 
        frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
      </td>
      <td>
        <video width="320" height="240" controls>
          <source src="https://54321anonymous.github.io/ICLR2025/versionA/89xTTczbv0E/gpt_univtg_title_threshold/gpt_univtg_title_threshold_demo.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
      </td>
      <td>
        <video width="320" height="240" controls>
          <source src="https://54321anonymous.github.io/ICLR2025/versionA/89xTTczbv0E/gpt_transformer_beamsearch_dp/gpt_transformer_beamsearch_dp_demo.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
      </td>
      <td>
        <video width="320" height="240" controls>
          <source src="https://54321anonymous.github.io/ICLR2025/versionA/89xTTczbv0E/GT/intro.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
      </td>
    </tr>
  </table>
</div>

<div style="text-align: center; margin: 0; padding: 0;">
  <img src="https://54321anonymous.github.io/ICLR2025/8x_all.png" style="max-width: 100%; height: auto; margin: 0 auto; display: block;">
</div>

---


## Section 2: Zero-shot Examples

**Descriptions**: In this section, we present some zero-shot examples with TeaserGen-PT and TeaserGen-LR.

### Ted Talks
<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>Input Video</strong></td>
    <td style="text-align: center;"><strong>TeaserGen-PT</strong></td>
    <td style="text-align: center;"><strong>TeaserGen-LR</strong></td>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/P6FORpg0KVo" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/gpt_title_zeroshot/ted/P6FORpg0KVo/P6FORpg0KVo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/zeroshot_dp/ted/P6FORpg0KVo/processed.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

<!-- ---

### NHK Documentaries

<table>
  <tr>
    <td style="text-align: center;"><strong>Input Video</strong></td>
    <td style="text-align: center;"><strong>TeaserGen-PT</strong></td>
    <td style="text-align: center;"><strong>TeaserGen-LR</strong></td>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/aansXcMqnNk" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/gpt_title_zeroshot/nhk/aansXcMqnNk/aansXcMqnNk.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/zeroshot_dp/nhk/aansXcMqnNk/processed.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table> -->

---

### BBC Documentaries
<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>Input Video</strong></td>
    <td style="text-align: center;"><strong>TeaserGen-PT</strong></td>
    <td style="text-align: center;"><strong>TeaserGen-LR</strong></td>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/wkQuOrsgVGY" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/gpt_title_zeroshot/bbc/wkQuOrsgVGY/wkQuOrsgVGY.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/zeroshot_dp/bbc/wkQuOrsgVGY/processed.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

### Netflix Documentaries

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>Input Video</strong></td>
    <td style="text-align: center;"><strong>TeaserGen-PT</strong></td>
    <td style="text-align: center;"><strong>TeaserGen-LR</strong></td>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/J8DGjUv-Vjc" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/gpt_title_zeroshot/netfliex/J8DGjUv-Vjc/J8DGjUv-Vjc.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/zeroshot_dp/netfliex/J8DGjUv-Vjc/processed.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

### Old Movies

<div style="overflow-x: auto;">
<table>
  <tr>
    <td style="text-align: center;"><strong>Input Video</strong></td>
    <td style="text-align: center;"><strong>TeaserGen-PT</strong></td>
    <td style="text-align: center;"><strong>TeaserGen-LR</strong></td>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/NzvjYjVDHIY" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/gpt_title_zeroshot/old_movie/NzvjYjVDHIY/NzvjYjVDHIY.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td>
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/zeroshot_dp/oldmovie/NzvjYjVDHIY/processed.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

## Section 3: Dataset Examples:

<!-- metadata, whole video link, three teaser track, title, text, teaser narration, also point out boundary point -->

**Descriptions**: In this section, we present two examples of our *DocumentaryNet*.

<div style="text-align: center; margin-bottom: 20px;">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/WfXRTZdyIe0?start=97" 
  frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
  <div style="overflow-x: auto; margin-bottom: 40px; flex: 1;">
    <strong>Teaser Music Track</strong><br>
    <audio controls style="margin-bottom: 20px;">
      <source src="https://54321anonymous.github.io/ICLR2025/WfXRTZdyIe0/music_intro.wav" type="audio/mpeg">
      Your browser does not support the audio element.
    </audio><br>
    <strong>Teaser Dialogue Track</strong><br>
    <audio controls style="margin-bottom: 20px;">
      <source src="https://54321anonymous.github.io/ICLR2025/WfXRTZdyIe0/dialog_intro.wav" type="audio/mpeg">
      Your browser does not support the audio element.
    </audio><br>
    <strong>Teaser Sound Effect Track</strong><br>
    <audio controls style="margin-bottom: 20px;">
      <source src="https://54321anonymous.github.io/ICLR2025/WfXRTZdyIe0/effect_intro.wav" type="audio/mpeg">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div>
    <strong>Title: Surviving Grand Canyon (Full Episode) | America's National Parks </strong><br>
    <strong>Tags:</strong><br>
        "national geographic", "nat geo", "natgeo", "animals", "wildlife", "science","explore",
        "discover","survival", "nature", "culture", "documentary", "perpetual planet nat geo",
        "photography", "full episode", "America's National Parks nat geo", "America's National Parks full episode",
        "America's National Parks", "National Parks", "Grand Canyon", "Grand Canyon National Park",
        "America's natural wonders", "Grand Canyon hiking", "National Parks adventure","Grand Canyon survival",
        "Nature conservation in Grand Canyon", "Wildlife in Grand Canyon"<br>
    <strong>Narration:</strong><br>
        [3.37->16.77]: The Grand Canyon, a chasm 277 miles long, even in winter, what appears Baron supports life.<br>
        [23.56->28.83]: A female mountain lion shelters from the cold, with her eight-month-old daughter.<br>
        [33.72->52.77]: She can't rest for long, with an extra mouth to feed, finding food is tough at the best of times.<br>
        [54.69->57.91]: But a good hunt can provide for a week or more.<br>
        [61.61->63.51]: Elk, are their main prey.<br>
        [65.42->67.5]: The canyon limits their escape routes.<br>
        [78.95->86.43]: Every day is a battle for survival, in one of America's grandest national parks.<br>
  </div>


---

<div style="text-align: center; margin-bottom: 20px;">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/Y8ePgiD8oiI?start=109" 
  frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
  <div style="overflow-x: auto; margin-bottom: 40px; flex: 1;">
    <strong>Teaser Music Track</strong><br>
    <audio controls style="margin-bottom: 20px;">
      <source src="https://54321anonymous.github.io/ICLR2025/Y8ePgiD8oiI/music_intro.wav" type="audio/mpeg">
      Your browser does not support the audio element.
    </audio><br>
    <strong>Teaser Dialogue Track</strong><br>
    <audio controls style="margin-bottom: 20px;">
      <source src="https://54321anonymous.github.io/ICLR2025/Y8ePgiD8oiI/dialog_intro.wav" type="audio/mpeg">
      Your browser does not support the audio element.
    </audio><br>
    <strong>Teaser Sound Effect Track</strong><br>
    <audio controls style="margin-bottom: 20px;">
      <source src="https://54321anonymous.github.io/ICLR2025/Y8ePgiD8oiI/effect_intro.wav" type="audio/mpeg">
      Your browser does not support the audio element.
    </audio>
  </div>
  <div>
    <strong>Title: Thai Cave Rescue (Full Episode) | Drain the Oceans </strong><br>
    <strong>Tags:</strong><br>
        "national geographic","nat geo","natgeo",
        "animals",
        "wildlife",
        "science",
        "explore",
        "discover",
        "survival",
        "nature",
        "culture",
        "documentary",
        "perpetual planet nat geo",
        "photography",
        "full episode",
        "Thai Cave Rescue",
        "Full Episode",
        "Drain the Oceans",
        "The Oceans",
        "First Accurate 3D Survey",
        "Thai Cave",
        "Insights",
        "Risky Mission",
        "Save the People",
        "Flooded Cave",
        "Rescue Mission",
        "Accurate 3D Survey",
        "Risk in Life",
        "Cave with Floods",
        "3D Survey"<br>
    <strong>Narration:</strong><br>
        [7.9->15.34]: For almost three weeks, 13 young lives hang in the balance.<br>
        [16.28->19.04]: We didn't want them to die and we didn't want to die ourselves.<br>
        [21.23->23.65]: Then, against all the odds.<br>
        [26.27->29.19]: Mission impossible has now become mission incredible.<br>
        [32.58->36.14]: One of the greatest rescues of all time.<br>
        [37.38->38.46]: It was like a miracle.<br>
        [42.15->43.61]: But questions were made.<br>
        [44.33->45.97]: How did the boys get trapped?<br>
        [47.27->50.05]: Why was it so hard to rescue them?<br>
        [51.01->61.39]: The prospect of a diving rescue was hideous from the outset.  And how was it done?  I don't think like this was ever happened before.  Everything was experimental.<br>
        [63.8->75.1]: Now, with unprecedented access,  a team of caveers conducts the first ever 3D scan of Tam Loay  to uncover the answers.<br>
        [77.71->80.81]: Reconstructing the cave in perfect detail.<br>
        [83.41->86.55]: We're going to be able to show what that passing really like.<br>
        [88.37->93.11]: Transforming our understanding of the Thai cave rescue.<br>
        [95.59->98.49]: If two of them came out alive, that would have been a good result.<br>
  </div>


---

## Section 4: More Examples:

**Descriptions**: In this section, we present additional teasers generated by TeaserGen and compare our model performance with baseline models.

<div style="overflow-x: auto;">
<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>Input Video</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>CLIP-IT</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>UniVTG</strong></th>
  </tr>
  <tr>
    <!-- <td style="border: 1px solid black; padding: 10px;"></td> -->
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/4xgPBb1xHYI?start=51" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/4xgPBb1xHYI/gpt_clip_rank/gpt_clip_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/4xgPBb1xHYI/gpt_univtg_rank/gpt_univtg_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-PT</strong></td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-LR</strong></td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/4xgPBb1xHYI/gpt_univtg_title_threshold/gpt_univtg_title_threshold_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/4xgPBb1xHYI/gpt_transformer_beamsearch_dp/gpt_transformer_beamsearch_dp_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>


---

<div style="overflow-x: auto;">
<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>Input Video</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>CLIP-IT</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>UniVTG</strong></th>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/ATvKJ_HftNs?start=119" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/ATvKJ_HftNs/gpt_clip_rank/gpt_clip_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/ATvKJ_HftNs/gpt_univtg_rank/gpt_univtg_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-PT</strong></td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-LR</strong></td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/ATvKJ_HftNs/gpt_univtg_title_threshold/gpt_univtg_title_threshold_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/ATvKJ_HftNs/gpt_transformer_beamsearch_dp/gpt_transformer_beamsearch_dp_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

<div style="overflow-x: auto;">
<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>Input Video</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>CLIP-IT</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>UniVTG</strong></th>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/Dy8ogOaKk4Y?start=171" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/Dy8ogOaKk4Y/gpt_clip_rank/gpt_clip_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/Dy8ogOaKk4Y/gpt_univtg_rank/gpt_univtg_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-PT</strong></td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-LR</strong></td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/Dy8ogOaKk4Y/gpt_univtg_title_threshold/gpt_univtg_title_threshold_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/Dy8ogOaKk4Y/gpt_transformer_beamsearch_dp/gpt_transformer_beamsearch_dp_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

<div style="overflow-x: auto;">
<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>Input Video</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>CLIP-IT</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>UniVTG</strong></th>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/khyuH_QfoWU?start=49" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/khyuH_QfoWU/gpt_clip_rank/gpt_clip_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/khyuH_QfoWU/gpt_univtg_rank/gpt_univtg_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-PT</strong></td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-LR</strong></td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/khyuH_QfoWU/gpt_univtg_title_threshold/gpt_univtg_title_threshold_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/khyuH_QfoWU/gpt_transformer_beamsearch_dp/gpt_transformer_beamsearch_dp_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

<div style="overflow-x: auto;">
<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>Input Video</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>CLIP-IT</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>UniVTG</strong></th>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/NsRj-9Y4fZ8?start=62" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/NsRj-9Y4fZ8/gpt_clip_rank/gpt_clip_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/NsRj-9Y4fZ8/gpt_univtg_rank/gpt_univtg_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-PT</strong></td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-LR</strong></td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/NsRj-9Y4fZ8/gpt_univtg_title_threshold/gpt_univtg_title_threshold_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/NsRj-9Y4fZ8/gpt_transformer_beamsearch_dp/gpt_transformer_beamsearch_dp_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

<div style="overflow-x: auto;">
<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>Input Video</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>CLIP-IT</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>UniVTG</strong></th>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/nTq1Sd9N7E8?start=43" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/nTq1Sd9N7E8/gpt_clip_rank/gpt_clip_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/nTq1Sd9N7E8/gpt_univtg_rank/gpt_univtg_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-PT</strong></td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-LR</strong></td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/nTq1Sd9N7E8/gpt_univtg_title_threshold/gpt_univtg_title_threshold_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/nTq1Sd9N7E8/gpt_transformer_beamsearch_dp/gpt_transformer_beamsearch_dp_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

<div style="overflow-x: auto;">
<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>Input Video</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>CLIP-IT</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>UniVTG</strong></th>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/psWarhwc4eY?start=84" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/psWarhwc4eY/gpt_clip_rank/gpt_clip_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/psWarhwc4eY/gpt_univtg_rank/gpt_univtg_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-PT</strong></td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-LR</strong></td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/psWarhwc4eY/gpt_univtg_title_threshold/gpt_univtg_title_threshold_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/psWarhwc4eY/gpt_transformer_beamsearch_dp/gpt_transformer_beamsearch_dp_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>


---

<div style="overflow-x: auto;">
<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>Input Video</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>CLIP-IT</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>UniVTG</strong></th>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/TlF8M_U7khI?start=95" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/TlF8M_U7khI/gpt_clip_rank/gpt_clip_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/TlF8M_U7khI/gpt_univtg_rank/gpt_univtg_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-PT</strong></td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-LR</strong></td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/TlF8M_U7khI/gpt_univtg_title_threshold/gpt_univtg_title_threshold_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/TlF8M_U7khI/gpt_transformer_beamsearch_dp/gpt_transformer_beamsearch_dp_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

---

<div style="overflow-x: auto;">
<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>Input Video</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>CLIP-IT</strong></th>
    <th style="border: 1px solid black; padding: 10px; text-align: center;"><strong>UniVTG</strong></th>
  </tr>
  <tr>
    <td>
      <iframe width="320" height="240" src="https://www.youtube.com/embed/vbl7QeqfE-Q?start=127" 
      frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/vbl7QeqfE-Q/gpt_clip_rank/gpt_clip_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/vbl7QeqfE-Q/gpt_univtg_rank/gpt_univtg_rank_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-PT</strong></td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;"><strong>TeaserGen-LR</strong></td>
  </tr>
  <tr>
   <td></td> <!-- This leaves the first column of the second row empty -->
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/vbl7QeqfE-Q/gpt_univtg_title_threshold/gpt_univtg_title_threshold_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <video width="320" height="240" controls>
        <source src="https://54321anonymous.github.io/ICLR2025/versionA/vbl7QeqfE-Q/gpt_transformer_beamsearch_dp/gpt_transformer_beamsearch_dp_demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>
</div>

*Disclaimer: We believe the use of materials in this demonstration falls under "fair use" for academic research purposes. We do not claim ownership of any copyrighted content included in this demo. Copyright holders who believe their content has been used inappropriately are encouraged to contact us for its removal.*