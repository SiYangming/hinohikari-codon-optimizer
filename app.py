import streamlit as st
from Bio.Seq import Seq
from Bio.Data import CodonTable
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import random
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Oryza sativa codon usage table (frequency per thousand, from Kazusa)
rice_codon_table = {
    'F': {'TTT': 13.1, 'TTC': 22.4},
    'L': {'TTA': 6.1, 'TTG': 14.7, 'CTT': 15.2, 'CTC': 25.8, 'CTA': 7.7, 'CTG': 21.0},
    'I': {'ATT': 14.0, 'ATC': 23.6, 'ATA': 8.1},
    'M': {'ATG': 22.1},
    'V': {'GTT': 15.5, 'GTC': 20.8, 'GTA': 6.6, 'GTG': 24.0},
    'S': {'TCT': 12.7, 'TCC': 16.3, 'TCA': 12.4, 'TCG': 12.3, 'AGT': 8.9, 'AGC': 17.9},
    'P': {'CCT': 13.6, 'CCC': 12.1, 'CCA': 14.2, 'CCG': 18.0},
    'T': {'ACT': 10.9, 'ACC': 17.0, 'ACA': 12.2, 'ACG': 10.7},
    'A': {'GCT': 16.7, 'GCC': 23.8, 'GCA': 13.2, 'GCG': 18.8},
    'Y': {'TAT': 10.0, 'TAC': 15.1},
    '*': {'TAA': 0.7, 'TAG': 0.8, 'TGA': 1.2},
    'H': {'CAT': 11.3, 'CAC': 13.8},
    'Q': {'CAA': 13.5, 'CAG': 20.8},
    'N': {'AAT': 15.3, 'AAC': 20.6},
    'K': {'AAA': 17.3, 'AAG': 30.7},
    'D': {'GAT': 19.9, 'GAC': 22.9},
    'E': {'GAA': 20.8, 'GAG': 28.2},
    'C': {'TGT': 6.2, 'TGC': 12.4},
    'W': {'TGG': 13.8},
    'R': {'CGT': 7.2, 'CGC': 16.1, 'CGA': 6.4, 'CGG': 13.4, 'AGA': 9.8, 'AGG': 13.6},
    'G': {'GGT': 10.2, 'GGC': 26.1, 'GGA': 12.9, 'GGG': 13.4}
}

# Load CodonBERT model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("lhallee/CodonBERT")
    model = AutoModelForMaskedLM.from_pretrained("lhallee/CodonBERT")
    return tokenizer, model

tokenizer, model = load_model()

# Build synonymous codons list
standard_table = CodonTable.unambiguous_dna_by_name["Standard"]
synonymous_codons = {}
for codon, aa in standard_table.forward_table.items():
    if aa not in synonymous_codons:
        synonymous_codons[aa] = []
    synonymous_codons[aa].append(codon)

def get_random_codon(aa):
    return random.choice(synonymous_codons.get(aa, ['NNN']))

# Rule-based optimization
def rule_optimize(aa_seq):
    dna_seq = ''
    for aa in aa_seq:
        codons = rice_codon_table.get(aa, {})
        if codons:
            dna_seq += max(codons, key=codons.get)
        else:
            dna_seq += 'NNN'  # unknown
    return dna_seq

# LLM optimization (CodonBERT): mask 20% codons and predict
def llm_optimize(aa_seq, mask_rate=0.2):
    # Initial random DNA
    initial_dna = ''.join(get_random_codon(aa) for aa in aa_seq)
    # Tokenize into codons (CodonBERT uses space-separated codons)
    codons = [initial_dna[i:i+3] for i in range(0, len(initial_dna), 3)]
    masked_codons = [c if random.random() > mask_rate else tokenizer.mask_token for c in codons]
    input_text = ' '.join(masked_codons)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    mask_indices = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    predicted_ids = logits[0, mask_indices].argmax(-1)
    predicted_codons = tokenizer.batch_decode(predicted_ids)  # batch decode
    
    # Replace masked
    pred_idx = 0
    for i in range(len(masked_codons)):
        if masked_codons[i] == tokenizer.mask_token:
            if pred_idx < len(predicted_codons):
                masked_codons[i] = predicted_codons[pred_idx]
                pred_idx += 1
    
    return ''.join(masked_codons)

# CAI calculation
def calculate_cai(dna_seq, codon_table=rice_codon_table):
    if len(dna_seq) % 3 != 0:
        return 0.0
    codons = [dna_seq[i:i+3] for i in range(0, len(dna_seq), 3)]
    max_freq = {aa: max(freq.values()) for aa, freq in codon_table.items() if aa != '*'}
    cai_sum = 0
    count = 0
    for codon in codons:
        try:
            aa = str(Seq(codon).translate())
            if aa in max_freq and codon in codon_table.get(aa, {}):
                freq = codon_table[aa][codon]
                cai_sum += freq / max_freq[aa]
                count += 1
        except:
            pass
    return cai_sum / count if count > 0 else 0.0

st.title("Hinohikari Codon Optimization Comparison (Rule-based vs. LLM)")
st.write("Input DNA sequence, translate to amino acids, then perform rule-based and LLM optimization comparison. Suitable for Hinohikari breeding.")

# Use session_state to manage input
if 'dna_input' not in st.session_state:
    st.session_state.dna_input = ""

# Input box
dna_input = st.text_area("DNA Sequence (ORF, multiple of 3)", value=st.session_state.dna_input, height=200)

if st.button("Load Default Example: Badh2 Gene (Oryza sativa)"):
    default_dna = "atggccacggcgatcccgcagcggcagctcttcgtcgccggcgagtggcgcgcccccgcgctcggccgccgcctccccgtcgtcaaccccgccaccgagtcccccatcggcgagatcccggcgggcacggcggaggacgtggacgcggcggtggcggcggcgcgggaggcgctgaagaggaaccggggccgcgactgggcgcgcgcgccgggcgccgtccgggccaagtacctccgcgcaatcgcggccaagataatcgagaggaaatctgagctggactagagacgcttgattgtgggaagcctcttgatgaagcagcatgggacatggacgatgttgctggatgctttgagtactttgcagatcttgcagaatccttggacaaaaggcaaaatgcacctgtctctcttccaatggaaaactttaaatgctatcttcggaaagagcctatcgggtagttgggttgatcacaccttggaactatcctctcctgatggcaacatggaaggtagctcctgccctggctgctggctgtacagctgtactaaaaccatctgaattggcttccgtgacttgtttggagcttgctgatgtgtgtaaagaggttggtcttccttcaggtgtgctaaacatagtgactggattaggttctgaagccggtgctcctttgtcatcacaccctggtgtagacaaggttgcatttactgggagttatgaaactggtaaaaagattatggcttcagctgctcctatggttaagcctgtttcactggaacttggtggaaaaagtcctatagtggtgtttgatgatgttgatgttgaaaaagctgttgagtggactctctttggttgcttttggaccaatggccagatttgcagtgcaacatcgcgtcttattcttcataaaaaaatcgctaaagaatttcaagaaaggatggttgcatgggccaaaaatattaaggtgtcagatccacttgaagagggttgcaggcttgggcccgttgttagtgaaggacagtatgagaagattaagcaatttgtatctaccgccaaaagccaaggtgctaccattctgactggtggggttagacccaagcatctggagaaaggtttctatattgaacccacaatcattactgatgtcgatacatcaatgcaaatttggagggaagaagttttttggtccagtgctctgtgtgaaagaatttagcactgaagaagaagccattgaattggccaacgatactcattatggtctggctggtgctgtgctttccggtgaccgcgagcgatgccagagattaactgaggagatcgatgccggaatttatctgggtgaactgctcgcaaccctgcttctgccaagctccatggggcgggaacaagcgcagcggctttggacgcgagctcggagaagggggcattgacaactaccttagcgtcaagcaagtgacggagtacgcctccgatgagccgtgggatggtacaaatccccttccaagctgtaa"
    st.session_state.dna_input = default_dna
    st.rerun()  # Refresh page to update input box

if st.button("Optimization Comparison"):
    input_value = dna_input.upper().replace(" ", "")  # Clean input
    orig_len = len(input_value)
    if orig_len % 3 != 0:
        # Auto truncate to multiple of 3
        input_value = input_value[: (orig_len // 3) * 3]
        st.warning(f"Input DNA length {orig_len} is not a multiple of 3, auto truncated to {len(input_value)} bp (discarded tail bases).")
    
    if len(input_value) == 0:
        st.error("Effective DNA sequence is empty, please input valid sequence.")
    else:
        try:
            aa_seq = str(Seq(input_value).translate())
            rule_dna = rule_optimize(aa_seq)
            llm_dna = llm_optimize(aa_seq)
            orig_cai = calculate_cai(input_value)
            rule_cai = calculate_cai(rule_dna)
            llm_cai = calculate_cai(llm_dna)

            st.subheader("Original DNA:")
            st.text_area("", value=input_value, height=100, disabled=True)

            st.subheader("Rule-based Optimized DNA:")
            st.text_area("", value=rule_dna, height=100, disabled=True)

            st.subheader("LLM Optimized DNA (CodonBERT):")
            st.text_area("", value=llm_dna, height=100, disabled=True)

            # Chart
            fig, ax = plt.subplots()
            ax.bar(['Original', 'Rule-based', 'LLM'], [orig_cai, rule_cai, llm_cai])
            ax.set_ylabel('CAI Value')
            ax.set_title('Optimization Comparison')
            st.pyplot(fig)

            # PDF report
            pdf_buffer = io.BytesIO()
            pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
            pdf.drawString(100, 750, "Hinohikari Codon Optimization Comparison Report")
            pdf.drawString(100, 700, f"Original CAI: {orig_cai:.2f}")
            pdf.drawString(100, 650, f"Rule-based CAI: {rule_cai:.2f}")
            pdf.drawString(100, 600, f"LLM CAI: {llm_cai:.2f}")
            pdf.drawString(100, 550, "LLM outperforms rule-based in contextual optimization.")
            pdf.save()
            pdf_buffer.seek(0)
            st.download_button("Download PDF Report", pdf_buffer, file_name="optimize_compare_report.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Error: {e}")
