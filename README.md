# VGTS

In the realm of historical manuscripts research, scholars dedicate countless hours to the meticulous task of identifying and recording new symbols discovered in the pages of historical texts. 

This process, while crucial to the progress of our understanding of history and culture, is currently carried out using rudimentary means that are both time-consuming and labor-intensive. 

Researchers, often working in resource-limited settings, must manually spot a new symbol, painstakingly document it on paper, and then collate these findings for future reference. This system, while diligent, is far from efficient, and can often lead to disarray and inaccuracies due to the human factor. For more details about our project, please refer to [Link](https://infinite-hwb.github.io/db.github.io/).

<img src="https://github.com/infinite-hwb/ots/blob/master/ST/Images/readme/db.png" width="633" >

*Fig 1. Manuscript Notes of a Historian Studying Dongba Texts: Newly Discovered Characters Categorized and Annotated*

Recognizing this pressing issue, we have embarked on an innovative project to revolutionize the field. We introduce a versatile text spotting model, an advanced tool designed to streamline the process of symbol documentation. 

Equipped with the capacity to handle multiple conditions, this groundbreaking tool will greatly enhance the efficiency and accuracy of symbol/character/text spotting in ancient books.

<img src="https://github.com/infinite-hwb/ots/blob/master/ST/Images/readme/2_1.png" width="800" >

*Fig 2. The Overall Framework of VGTS*

Fig 2 illustrates the model's framework.

<img src="https://github.com/infinite-hwb/ots/blob/master/ST/Images/readme/2_2.png" width="800" >

*Fig 3. Visualization Results on the DBH datasets*

Fig 3 shows the VGTS model's visual results on the DBH dataset. Here, green boxes indicate categories identified during training ('Base'), while blue boxes highlight categories not seen in the training phase ('Novel').

Furthermore, our model demonstrates impressive performance on additional datasets. For instance, Fig 4 presents results on the Tripitaka Koreana in Han dataset, showcasing ancient Chinese Buddhist scriptures, while Fig 5 exhibits performance on the Egyptian Hieroglyph Dataset.

<img src="https://github.com/infinite-hwb/ots/blob/master/ST/Images/readme/2_3.png" width="800" >

*Fig 4. Performance on the Tripitaka Koreana in Han Dataset, Illustrating Ancient Chinese Buddhist Scriptures*

<img src="https://github.com/infinite-hwb/ots/blob/master/ST/Images/readme/2_4.png" width="800" >

*Fig 5.  Performance on the Egyptian Hieroglyph Dataset, Showcasing Ancient Egyptian Hieroglyphs*

**Acknowledgments**

This project has been greatly influenced by the foundational work of numerous researchers in the field. In particular, we would like to express our gratitude to:

•	Osokin, Sumin, and Lomakin, for their paper "Os2d: One-stage one-shot object detection by matching anchor features", presented at the European Conference on Computer Vision (ECCV), 2020, pages 635-652. Their work provided invaluable insights and methodologies that significantly influenced the development of our code.

•	Jaderberg, Simonyan, and Zisserman, for their groundbreaking research presented in "Spatial transformer networks", published in Advances in Neural Information Processing Systems, vol. 2, pages 2017–2025, 2015. Their innovative ideas have formed a substantial part of the backbone of our project.

We also want to extend our appreciation to all other researchers whose work contributes directly or indirectly to this project. Their pioneering contributions to the field have laid the groundwork for our exploration and development.

This project would not have been possible without their valuable contributions, and for that, we are deeply grateful.
