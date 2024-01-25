{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "6c5cc776-c427-4bab-9f61-d0d1bc71cbed",
   "metadata": {},
   "source": [
    "## Integrating ChatGPT"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "34e1b7bd-2443-4ae6-816a-002edba31897",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "pip install openai"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "b23d0c00-fe8c-4860-ab4b-0d08eb4da62f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "What type of chatbot would you like to create?\n",
      " informational\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Your new assistant is ready! Type 'quit()' to end the session.\n"
     ]
    },
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "You:  when was stable diffusion created?\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Assistant: Stable diffusion characterizes a phenomenon, rather than being a specific technology or invention with a single creation date. Diffusion refers to the spread of innovations, ideas, or technologies from one group or society to another. The concept of diffusion can be traced back to the work of anthropologist Franz Boas in the late 19th and early 20th centuries. Boas examined the diffusion of cultural traits and practices among indigenous groups in North America. Since then, the study of diffusion has been further developed by various social scientists and has become a fundamental concept in fields such as anthropology, sociology, and innovation studies. So, stable diffusion can be considered as an ongoing process rather than something that was created at a specific point in time.\n",
      "\n"
     ]
    },
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "You:  quit()\n"
     ]
    }
   ],
   "source": [
    "from openai import OpenAI\n",
    "import os\n",
    "import pandas as pd\n",
    "import time\n",
    "\n",
    "client = OpenAI(api_key = 'your-api-key')\n",
    "\n",
    "messages = []\n",
    "system_msg = input(\"What type of chatbot would you like to create?\\n\")\n",
    "messages.append({\"role\": \"system\", \"content\": system_msg})\n",
    "\n",
    "print(\"Your new assistant is ready! Type 'quit()' to end the session.\")\n",
    "\n",
    "# Loop for chat interaction\n",
    "while True:\n",
    "    user_message = input(\"You: \")\n",
    "    if user_message == \"quit()\":\n",
    "        break\n",
    "\n",
    "    messages.append({\"role\": \"user\", \"content\": user_message})\n",
    "    \n",
    "    try:\n",
    "        response = client.chat.completions.create(\n",
    "            model=\"gpt-3.5-turbo\",\n",
    "            messages=messages\n",
    "        )\n",
    "        reply = response.choices[0].message.content\n",
    "        messages.append({\"role\": \"assistant\", \"content\": reply})\n",
    "        print(\"\\nAssistant: \" + reply + \"\\n\")\n",
    "    except Exception as e:\n",
    "        print(\"An error occurred: \", e)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ea044a1e-f1b6-4be8-bf2a-51bcb0c8dd18",
   "metadata": {},
   "source": [
    "### ChatGPT fails to answer questions out of its scope\n",
    "Above I integrated ChatGPT-3.5 to the python notebook and asked it a question about Stable diffusion.\n",
    "Since, GPT-3.5 has data upto 2021 and stable diffusion was released in 2022, the answer output by GPT-3.5 was hallucinating in nature.\n",
    "\n",
    "To solve this problem we can feed ChatGPT with the relevant information after collecting the relevant information."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f640b069-6840-46fa-85f0-b68b4772ae20",
   "metadata": {},
   "source": [
    "# Implementing RAG using llama-index"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b1701d1e-d22c-4a3e-a12d-e7bf4af6a332",
   "metadata": {},
   "source": [
    "I scraped data from wikipedia related to Stable Diffusion and loaded that data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "e049d234-b3dd-402c-a205-b7cdb0e47dce",
   "metadata": {},
   "outputs": [],
   "source": [
    "# pip install llama-index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "c1e4b525-e110-4889-b76a-79ee40200d15",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from llama_index import VectorStoreIndex, SimpleDirectoryReader\n",
    "OPENAI_API_KEY = 'your-api-key'\n",
    "\n",
    "# Set the OpenAI API key\n",
    "os.environ[\"OPENAI_API_KEY\"] = OPENAI_API_KEY\n",
    "\n",
    "documents = SimpleDirectoryReader(r\"C:\\Users\\user\\OneDrive\\Desktop\\Data\\stable_diffusion\").load_data()\n",
    "index = VectorStoreIndex.from_documents(documents)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d4462123-c7a5-4ffb-9ce0-0811b7cfe688",
   "metadata": {},
   "source": [
    "In the above code cell, the loaded data was being vectorized so that it can be made available to ChatGPT in readable format."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f0b5554a-5e3a-468f-8b27-72bebedb75f7",
   "metadata": {},
   "source": [
    "### Final Step\n",
    "Indexing the relevant data that matches the context of the query. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "63362b5f-eb86-47c8-99fa-0bfbf2cf3cd6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stable Diffusion was released in 2022.\n"
     ]
    }
   ],
   "source": [
    "query_engine = index.as_query_engine()\n",
    "response = query_engine.query(\"When was stable diffusion released?\")\n",
    "print(response)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9bd473b2-c6cc-4de7-a9d7-360215a689a5",
   "metadata": {},
   "source": [
    "In this case, Indexing whatever ChatGPT can find about stable diffusion from data it was fed.\n",
    "From the response you can see, ChatGPT is giving a much better answer than before for the SAME question."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
