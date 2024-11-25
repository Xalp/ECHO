# ECHO
Official homepage for "s**E**lf-Harmonized **C**hain of t**HO**ught" https://www.arxiv.org/abs/2409.04057 .

[_UPDATE_ **Nov-25-2024** : We update parallel inference code to speed up the inference speed by x20!!! ]

[_UPDATE_ Sept-19-2024 : We found the log is not upload due to wrong .gitignore ; Re-upload the log, you can now create your own demonstrations!!! ]

[_UPDATE_ Sept-16-2024 : We update the code to fit to the lastest OPENAI version, as requested by issue. We also share the log and demo to save your effort.]

[_UPDATE_ Feb-16-2024 : First Submission.]

<div align="center">
<img src="echo.webp" width="200" height="150">
</div>

## Get Started

Please install the latest __openai__ and __torch__

Set your API:

```
export OPENAI_API_KEY=(YOUR OPENAI API KEY)
```

Similarly, you should set MISTRAL_API_KEY if you want to use the MISTRAL models.

### Step to reproduce main experiment results

(Optional) **Step 0: Log Creation** 

We have created the log for you using Zero-Shot CoT. This serves as an initialization of our method. If you are using other models, you may consider running this.

Note that this is not compulsory, as the demo selection will only depend on the question, not the rationale. You can skip this step and directly regenerate the rationale using your model.
```
source run/crate_log.sh
```

**Step 1: DEMO creation** 

We follow Auto-CoT to use the clustering, and then select one question from each cluster.
As k=max requires maximum number of demonstration allowed by context length, we generate a number of 8 to 32 demonstrations for each dataset. If you are not running k=max case, you can generate only 8 demonstrations.

```
source run/create_demos.sh
```

**Step 2: RUN ECHO~!**

Now you can run ECHO with
```
source run/run_echo.sh
```

This script creates the demo with ECHO;

**Step 3: RUN ECHO (k=max)**

```
source run_echo_max.sh
```

This script creates the demo with ECHO(k=max); This script will test the maximum number of demonstrations allowed by context window size iteratively, starting from 32 to 8;

**Step 4: run inference**

We attached the inference code after run_echo.sh and run_echo_max.sh
If you want to test Auto-CoT, please use the demo: 
```
demos/{dataset}_{model_name}
```
If you want to test manual prompt (from Few-shot-CoT), please use the demo: 
```
demos/{dataset}_manual
```
We suggest **T=4** for optimal performance. However, we found that an easier and less diverse dataset may require less iteration.

You can replace "singleeq" to any other datasets we included: "aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters"

We also include the code for inferencing Mistral API.

If you have any question, please consider raise an issue or directly email Ziqi.

## Some Intuition

There are multiple intuitions which I'd like to share after finishing this paper:

(1) The unified demonstrations will better match the case in the pre-training data of the model, where contexts from the same piece are mutually relevant and consistent.

(2) Cognitive Load Theory (by John Sweller): learning is most effective when the cognitive load on working memory is minimized. If all demonstrations are coherent, it is easier to learn the pattern and follow both humans and models.

(3) You can also explain this with the Entropy Theory: unified demos reduce the information entropy (disorder and uncertainty), thus increasing the predictability.

## Acknowledgement

This repo is built on repos of Auto-CoT and Zero-shot-CoT.

## Citation

To cite our paper, please include the following bibtex:

```
@misc{jin2024selfharmonizedchainthought,
      title={Self-Harmonized Chain of Thought}, 
      author={Ziqi Jin and Wei Lu},
      year={2024},
      eprint={2409.04057},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.04057}, 
}
```
