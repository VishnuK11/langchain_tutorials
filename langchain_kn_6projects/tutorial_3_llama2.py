from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import CTransformers
import streamlit as st



# Function to get Response from LLama 2 model
def get_model_response(input_text, no_words, blog_style):
    llm = CTransformers(model='langchain_kn_6projects/models/llama-2-7b-chat.Q3_K_S.gguf',
                        model_type='llama',
                        config={'max_new_tokens':256,
                                'temperature':0.3})

    # Prompt Template
    template = f"""
    Write a blog for {blog_style} job profile for a topic {input_text}     within {no_words} words.
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    chain = prompt | llm
    response = chain.invoke({"blog_style":blog_style,
                                  "input_text":input_text,
                                  "no_words":no_words})
    print(response)
    return response



if __name__ == '__main__':

    st.set_page_config(page_title="Generate Blogs",
                       page_icon=':)',
                       layout='centered',
                       initial_sidebar_state='collapsed')
    st.header("Generate Blogs :D")
    input_text = st.text_input("Provide the Blog Topic")

    col1,col2 = st.columns([5,5])

    with col1:
        no_words = st.text_input("No. of Words")
    with col2:
        blog_style = st.selectbox("Writing the blog for",
                                  ("Researchers", "Data Scientists", "Common Man"), index=0)

    submit = st.button("Generate")

    if submit:
        st.write(get_model_response(input_text, no_words, blog_style))

    # st.write(get_model_response("Large Language Model", 300, "Researcher"))


