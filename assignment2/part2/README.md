
## Installation

For this part of the assignment, you need to install the `clip` package from OpenAI. You can install it by running the following command:

(Make sure you first navigate to the `part2` directory.)

```bash
sbatch install_clip.sh
```

* Note: This assumes that you are using `conda` environment named `dl2022`. If you are using a different environment, you need to change the `install_clip.sh` script accordingly.
* Once you run the script, the slurm output file should end with something like:
    ```sh
    Successfully built clip
    Installing collected packages: clip
    Successfully installed clip-1.0
    CLIP available models: ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    ```

Please post on Piazza in case you face any issues in this step.

## Part 2.1: Zero-shot CLIP


**Overview**

As explained in the PDF, in this part, what you need to do is keep CLIP frozen, pass your image(s) and class label(s) through CLIP and rank the classes based on the CLIP scores.

In this part, your goal is to fill the script `clipzs.py` to achieve this. First, go over the script carefully. Start from the `main()` function and try to understand what each line is doing.

> Note: Wherever you see the following snippet, you need to add your code there:
> ```python
>    #######################
>    # PUT YOUR CODE HERE  #
>    #######################
>    # :
>    # This part will have TODOs and hints
>    # :
>    #######################
>    # END OF YOUR CODE    #
>    #######################
> ```


**Key steps**

1. First, you need to complete the class `ZeroshotCLIP`. Speficially, finish functions:
    * `precompute_text_features()`: This should compute CLIP features for each class label in the dataset. Please see the function for more details.
    * `model_inference()`: This function takes an image and returns logit scores for all classes in the dataset. Please see the function for more details.
2. Second, you need to complete the `main()` function by adding an inference loop. Please see the function for more details.

**How to run the code?**

* If you want to use the interactive terminal on the login node itself to run the code, then follow:
    ```sh
    # Activate the conda environment
    module purge
    module load 2021
    module load Anaconda3/2021.05
    source activate dl2022

    # Run the script
    python $code_dir/clipzs.py --dataset cifar10 --split test
    ```
    This will run zero-shot evaluation of CLIP on test set of CIFAR-10 dataset.

* If you want to run it through a jobscript, we provide a sample:
    ```sh
    sbatch run_clipzs.sh
    ```
    To run variants, please see the arguments in `clipzs.py` and run accordingly.

The results from this part serve as a baseline for the next section.

Once you have the base code, you can modify the existing jobscript or write your own code/job scripts to run experiments/visualize results.

## Part 2.2: Visual Prompting with CLIP

**Overview**

As explained in the PDF, the goal of visual prompting is to modify the input (image) in a way that helps steer the model to perform better at a given task, e.g. image classification.

In this case, our modification of the image is *additive*, i.e., for a given input image $x$, we modify it as follows: $x' = x + v$
where $v$ is a (learnable) variable that is the *same* for each image in the dataset, and specific to a given task. It is easier to think of $v$ as a (frame) padding over the image as shown in the Fig below. You want to change $v$ (by training) s.t. the model's performance improves. This approach is called *visual prompting*. Note that we keep the text inputs (class label) unchanged, i.e., same as in part 1.

<img width="800" alt="image" src="https://user-images.githubusercontent.com/19412343/201550242-1076d0ae-e848-4b58-af58-5eef34963e90.png">

**Visual prompts**

There can be various ways of defining $v$, e.g., padding around the image. According to the assignment PDF, you need to first implement three types of visual prompts:

1. *Single pixel patch*: Fill in the class `FixedPatchPrompter` in `vp.py`. Note that the patch size is a parameter to this class - for this part, you need to use a patch size of 1. See (a) in figure below.
2. *Random pixel patch*: Fill in the class `RandomPatchPrompter` in `vp.py`. Note that the patch size is a parameter to this class - for this part, you need to use a patch size of 1. See (b) in figure below.
3. *Padding*: Fill in the class `PadPrompter` in `vp.py`. Note that the padding size is a parameter to this class - for this part, you need to use a padding size of 30. See (c) in figure below.

<img width="800" alt="image" src="https://user-images.githubusercontent.com/8458550/201727624-389295ac-8fa6-4e95-8972-5297f3191010.png">

**Visual Prompt Tuning (VPT) Model**

Next, fill in parts of `vpt_model.py`. Specifically, you need to fill in the following functions:
1. In the `__init__()` function, add code to pre-compute text features (similar to part 2.1)
2. Then add the `__forward__()` function which does the following:
    * First, attaches the visual prompt to the input image
    * Then, does the usual forward pass of CLIP (similar to part 2.1)

**VPT Training**

Now that you have the visual prompters and the model ready, you can write the training loop. This is done in `learner.py` script.

1. First, in the function `train_one_epoch()`, fill in the code for a single batch. See file for more details.
2. Next, in the funtion `evaluate()`, fill in the code for evaluation. See file for more details.

**How to run the code?**

Running `main.py` will start the training process. You can use `--help` to see the available options.
Again we provide a sample job script (`run_clipvp.job`) that runs this part with default arguments for CIFAR-10.

Once you have the base code, you can write your own code/job scripts to run experiments/visualize results.

## Part 2.3: Robustness to Noise

In this section, the robustness of your learnt prompts is evaluated against distributional shifts. To do this,
you can add Gaussian noise to the test set of each dataset and observe whether there is a significant drop in performance.

1. First, fill the missing code snippets the function ``__call__()`` of the class ``AddGaussianNoise`` in the file ``dataset.py``:
    - This function adds a Gaussian noise $𝒩(\mu, \sigma^2)$ to a batch of images.

2. Call robustness.py with the argument ``--test_noise`` to add noise to the test set’s images, and the argument ``--resume`` to load the best performant checkpoint as:

```
python robustness.py --dataset {cifar10/cifar100} --resume ./{path_to_checkpoint_folder}/model_best.pth.tar --evaluate --test_noise
```

You also can evaluate your model performance on without noise with:
```
python robustness.py --dataset {cifar10/cifar100} --resume ./{path_to_checkpoint_folder}/model_best.pth.tar --evaluate
```


## Part 2.4: Cross-dataset Evaluation

In this section, you evaluate the effectiveness of your learnt visual prompts on the combination of CIFAR-10 and CIFAR-100. More specifically, you should compare CLIP’s performance when predicting a class out of both datasets’ labels (i.e. perform a 110-way classification, as the two sets of classes are mutually exclusive) with its performance on each dataset individually from the previous questions.

1. First, fill the missing code snippets in file ``cross_dataset.py``.

2. Call cross_dataset.py with argument ``--evaluate`` for evaluation mode, and argument ``--resume`` to load the best performing checkpoint as:

```
python cross_dataset.py --dataset cifar10 --resume ./{path_to_checkpoint_folder}/model_best.pth.tar --evaluate
```
```
python cross_dataset.py --dataset cifar100 --resume ./{path_to_checkpoint_folder}/model_best.pth.tar --evaluate
```



