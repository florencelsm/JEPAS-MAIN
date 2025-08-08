1 - Initially we thought generating multiple target blocks from within the Bounding Box was a good idea. But it appears
   that the Bounding Boxes are sometimes NOT at the correct poistion, or they are too small to generate target patches.

   Solution: 
        - When the bounding box is small use all of it as ONE Target Block
        - When Bounding Box is large, generate MULTIPLE Target Blocks from within the Bounding Box
        - Bounding Boxes that are not at correct position are not a lot so maybe we can just ignore this issue.

    TODO:
        - Load the Bounding Box in dataset.py
    
2 - I am 99.372% sure that the context blocks and target blocks generate is incorrect.

    TODO:
        - I (Florence**2) need to be 10000% sure that the context blocks and target blocks generatation is correct 
          because my project is highly dependent on it.