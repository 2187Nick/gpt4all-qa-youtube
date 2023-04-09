# gpt4all-qa-youtube
🚀 Query any youtube video transcript with your AI model
![green1](https://user-images.githubusercontent.com/75052782/230746190-5c499781-c539-4eef-a164-853c392fced7.jpg) 

> How to Use it 🦖:

```
  Enter any youtube video.
  Enter your question.
  Press Submit.
  
  Output0: Gives the answer based on data from the video.
  Output1: Gives the answer based on the data from your model.
  
  
  This uses Chroma as the vector store.
  
  *** This can take up to 60 seconds. Depending on your setup.
  
```
> Settings 👷‍:

```bash

  1. Select model. Default is set to "models" folder.
       Increasing this will give longer responses.

  2. Response length. Default is set to 30.
       Increasing this will give longer responses.
      
  3. Temperature. Default is .3
       Increasing this number will create more random results.
      
  4.  Embeddings.  Default is all-MiniLM-L6-v2.
  
  *** Use python <= 3.10
        
       
 ```
 
 > Run Script 👷‍:
``` bash

    python cli-version.py
    or
    python gradio-version.py


```
