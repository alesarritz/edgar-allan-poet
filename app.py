import streamlit as st
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = "edgar_allan_poet_model"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

tokenizer, model = load_model()

# Helpers
def split_title_and_poem(text, max_lines=12):
    lines = text.strip().splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    if not lines:
        return "Untitled", ""
    title = lines[0]
    filtered = []
    for line in lines[1:]:
        if line.lower() == title.lower():
            continue
        if line.strip() in {".", ":", "…"} or line.isspace():
            continue
        filtered.append(line)
    poem_body = "\n".join(filtered[:max_lines])
    return title, poem_body

def generate_poem(prompt, max_length=150, temperature=0.8, top_k=30):
    input_text = f"Write a poem about {prompt}.\n"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)

    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    raw_poem = generated_text[len(input_text):].strip()
    title, poem = split_title_and_poem(raw_poem)
    return title, poem

# Streamlit UI
st.title("Edgar Allan Poet")
st.subheader("Generate a poem from any idea or theme")

user_prompt = st.text_input("Hello Edgar, write a poem about", placeholder="e.g. a lonely winter evening")

if st.button("Generate Poem") and user_prompt:
    with st.spinner("Composing verse..."):
        title, poem = generate_poem(user_prompt)
    st.markdown(f"### {title}")
    st.text(poem)
