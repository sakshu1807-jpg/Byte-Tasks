## Task 2 : The Manga Task

This README file has the description for the Manga Task.
Honestly, by performing this task (even though I couldn't fine tune it ) I learned a a lot about the Vision Language Models (VLM) and also about the OCR.
It's types like EasyOCR, or the LLM approach. Also, got to learn about the LORA (Low Ranked Adapters), and how they efficiently reduces the memory head of the pre trained model.
First my approach was to already split the train test validation data with their respective labels to avoid any kind of data or image mix up. 
So, i first created a train_test_split.py file script to move the sequences to their respective types of data with the help of sequences.json file. 

then I used the open weight model from hugging face named "Qwen/Qwen2-VL-2B-Instruct". I first ran this model without fine tuning on Google Colab to see how it performs and it performed poorly.
When compared with the score.py it gave a score around 0.1 or 0.2. 
When I tried to fine tune it, but got various errors and the enough time wasn't left. So, I directly called the model on the test data and then pasted the final rpedictions in the sample_submissions.json file.

In future, I will surely look to upgrade myself towards the open weight and open source models... 
Thank you, for such a challenging task 😊
