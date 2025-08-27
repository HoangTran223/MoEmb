#!/usr/bin/env python3
"""
Compute global alignment matrix following the paper's offline phase:
1) Tokenize a large corpus (one file) separately with teacher and student tokenizers.
2) Train FastText on each tokenized corpus (token sequences, tokens separated by spaces).
3) Extract token embeddings for selected vocab (optionally top-k) and compute cost = 1 - cosine.
4) Solve OT (Sinkhorn) to obtain global alignment matrix and save it.

Usage example:
  python create_global_alignment_offline.py \
    --teacher-model bert-base-uncased \
    --student-model sentence-transformers/all-MiniLM-L6-v2 \
    --corpus-path /path/to/enwiki.txt \
    --output-path ./global_alignment.npy \
    --top-k 50000

Note: requires `transformers`, `fasttext`, `POT`, `tqdm`, `numpy`.
"""

import os
import sys
import argparse
import tempfile
import logging
from collections import Counter

import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer
import requests
import shutil
import bz2
import subprocess
import threading
import time
import itertools

try:
    import fasttext
except Exception as e:
    print("Error: fasttext not installed. Install with: pip install fasttext")
    raise

try:
    import ot
except Exception as e:
    print("Error: POT (python Optimal Transport) not installed. Install with: pip install POT")
    raise


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def tokenize_corpus_to_file(tokenizer, corpus_path, out_path, max_lines=None, desc=None):
    """Tokenize corpus lines with a HuggingFace tokenizer and write tokenized lines (tokens separated by spaces).
    Returns token frequency counter.
    """
    counter = Counter()
    total = 0
    with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as inf, \
         open(out_path, 'w', encoding='utf-8') as outf:
        for line in tqdm(inf, desc=desc or f"Tokenizing {os.path.basename(out_path)}", unit='lines'):
            if max_lines is not None and total >= max_lines:
                break
            text = line.strip()
            if not text:
                continue
            # Tokenize without adding special tokens, get token ids then convert to tokens for stability
            try:
                ids = tokenizer(text, add_special_tokens=False)['input_ids']
                tokens = tokenizer.convert_ids_to_tokens(ids)
            except Exception:
                # fallback to tokenizer.tokenize
                tokens = tokenizer.tokenize(text)

            # Clean token markers commonly used by BPE/SP: remove leading special markers so tokens are more "wordlike"
            cleaned = []
            for t in tokens:
                ct = t.replace('Ġ', '').replace('▁', '')
                ct = ct.strip()
                if ct:
                    cleaned.append(ct)
                    counter[ct] += 1
            if len(cleaned) == 0:
                # write an empty token placeholder to keep sentence count consistent
                outf.write('\n')
            else:
                outf.write(' '.join(cleaned) + '\n')
            total += 1
    return counter


def download_file(url, out_path):
    """Download a file by streaming to out_path."""
    logging.info(f"Downloading {url} -> {out_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0) or 0)
        with open(out_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc='Downloading') as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    logging.info("Download finished")


def extract_wiki_bz2_to_text(bz2_path, out_path, max_pages=None):
    """Try to extract plain text from a Wikipedia XML bz2 dump.
    Prefer gensim's WikiCorpus if available. If not available, instruct user to install WikiExtractor
    or gensim.
    """
    logging.info(f"Extracting wiki dump {bz2_path} -> {out_path}")
    # Try a couple of import paths for WikiCorpus (gensim reorganizations differ across versions)
    WikiCorpus = None
    import_err = None
    try:
        try:
            # preferred location in many gensim versions
            from gensim.corpora.wikicorpus import WikiCorpus as _WC
            WikiCorpus = _WC
        except Exception:
            # fallback older import path
            try:
                from gensim.corpora import WikiCorpus as _WC
                WikiCorpus = _WC
            except Exception as e:
                import_err = e
                WikiCorpus = None
    except Exception as e:
        import_err = e
        WikiCorpus = None

    # If gensim's WikiCorpus is available, use it (streaming, memory-efficient)
    if WikiCorpus is not None:
        try:
            wiki = WikiCorpus(bz2_path, lemmatize=False, dictionary={})
            with open(out_path, 'w', encoding='utf-8') as fout:
                for i, text in enumerate(tqdm(wiki.get_texts(), desc='Extracting wiki texts')):
                    fout.write(' '.join(text) + '\n')
                    if max_pages is not None and (i + 1) >= max_pages:
                        break
            logging.info("Extraction finished (gensim WikiCorpus)")
            return
        except Exception as e:
            logging.warning(f"gensim.WikiCorpus initialization/extraction failed: {e}")
            import_err = e

    # If we reach here, gensim's WikiCorpus was not usable. Try a simple streaming fallback
    # This fallback scans the bz2 XML and extracts text between <text>...</text> tags. It's
    # not as feature-complete as WikiExtractor but works for producing a plain-text corpus
    # suitable for FastText training when gensim/WikiExtractor are unavailable or broken.
    logging.info("Falling back to simple bz2 XML streaming parser for enwiki (may be slower)")
    try:
        with bz2.open(bz2_path, 'rt', encoding='utf-8', errors='ignore') as inf, \
             open(out_path, 'w', encoding='utf-8') as fout:
            in_text = False
            buffer_lines = []
            pages = 0
            for raw in tqdm(inf, desc='Streaming enwiki XML'):
                line = raw.strip()
                if not in_text:
                    # detect <text ...> tag
                    if '<text' in line:
                        # If opening and closing on same line
                        if '</text>' in line:
                            # quick extract between tags
                            start = line.find('>') + 1
                            end = line.rfind('</text>')
                            content = line[start:end].strip()
                            if content:
                                fout.write(content.replace('\n', ' ') + '\n')
                                pages += 1
                                if max_pages is not None and pages >= max_pages:
                                    break
                        else:
                            in_text = True
                            # take suffix after opening tag
                            start = line.find('>') + 1
                            suffix = line[start:].strip()
                            if suffix:
                                buffer_lines.append(suffix)
                else:
                    # inside text block
                    if '</text>' in line:
                        # write buffered content + prefix before closing tag
                        end = line.find('</text>')
                        prefix = line[:end].strip()
                        if prefix:
                            buffer_lines.append(prefix)
                        # join buffer and write as one document line
                        content = ' '.join(buffer_lines)
                        if content:
                            fout.write(content.replace('\n', ' ') + '\n')
                            pages += 1
                        buffer_lines = []
                        in_text = False
                        if max_pages is not None and pages >= max_pages:
                            break
                    else:
                        buffer_lines.append(line)
        logging.info("Extraction finished (bz2 streaming fallback)")
        return
    except Exception as e:
        logging.error(f"Fallback bz2 streaming extraction failed: {e}")
        if import_err is not None:
            logging.error(f"Earlier gensim import/extract error: {import_err}")
        logging.error("Install gensim or WikiExtractor, or provide --corpus-path to a plain text corpus.")
        raise


def stream_tokenize_wiki_bz2(bz2_path, teacher_tokenizer, student_tokenizer, teacher_out_path, student_out_path, max_pages=None):
    """Stream-extract wiki bz2 and write tokenized lines directly for teacher and student tokenizers.
    Returns two Counters (teacher_counter, student_counter).
    This avoids writing a large intermediate extracted text file.
    """
    logging.info(f"Streaming-extract + tokenize {bz2_path} -> {teacher_out_path} and {student_out_path}")
    teacher_counter = Counter()
    student_counter = Counter()
    total_pages = 0
    with bz2.open(bz2_path, 'rt', encoding='utf-8', errors='ignore') as inf, \
         open(teacher_out_path, 'w', encoding='utf-8') as t_out, \
         open(student_out_path, 'w', encoding='utf-8') as s_out:
        in_text = False
        buffer_lines = []
        for raw in tqdm(inf, desc='Streaming enwiki XML and tokenizing'):
            line = raw.strip()
            if not in_text:
                if '<text' in line:
                    if '</text>' in line:
                        start = line.find('>') + 1
                        end = line.rfind('</text>')
                        content = line[start:end].strip()
                        if content:
                            # tokenize and write both
                            _write_tokenized_line(content, teacher_tokenizer, t_out, teacher_counter)
                            _write_tokenized_line(content, student_tokenizer, s_out, student_counter)
                            total_pages += 1
                            if max_pages is not None and total_pages >= max_pages:
                                break
                    else:
                        in_text = True
                        start = line.find('>') + 1
                        suffix = line[start:].strip()
                        if suffix:
                            buffer_lines.append(suffix)
            else:
                if '</text>' in line:
                    end = line.find('</text>')
                    prefix = line[:end].strip()
                    if prefix:
                        buffer_lines.append(prefix)
                    content = ' '.join(buffer_lines)
                    if content:
                        _write_tokenized_line(content, teacher_tokenizer, t_out, teacher_counter)
                        _write_tokenized_line(content, student_tokenizer, s_out, student_counter)
                        total_pages += 1
                    buffer_lines = []
                    in_text = False
                    if max_pages is not None and total_pages >= max_pages:
                        break
                else:
                    buffer_lines.append(line)
    logging.info(f"Streaming tokenization finished: wrote {total_pages} documents")
    return teacher_counter, student_counter


def _write_tokenized_line(text, tokenizer, out_file, counter):
    try:
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        tokens = tokenizer.tokenize(text)
    cleaned = []
    for t in tokens:
        ct = t.replace('Ġ', '').replace('▁', '').strip()
        if ct:
            cleaned.append(ct)
            counter[ct] += 1
    if len(cleaned) == 0:
        out_file.write('\n')
    else:
        out_file.write(' '.join(cleaned) + '\n')


def _run_with_spinner(target, args=(), kwargs=None, desc='working'):
    """Run target(*args, **kwargs) in a background thread and display a small tqdm spinner until done."""
    if kwargs is None:
        kwargs = {}
    result = {}

    def _wrap():
        try:
            result['val'] = target(*args, **kwargs)
        except Exception as e:
            result['exc'] = e

    t = threading.Thread(target=_wrap)
    t.start()
    spinner = itertools.cycle(['|', '/', '-', '\\'])
    with tqdm(total=0, bar_format='{desc} {elapsed}', desc=desc) as p:
        while t.is_alive():
            p.set_description(f"{desc} {next(spinner)}")
            time.sleep(0.5)

    if 'exc' in result:
        raise result['exc']
    return result.get('val')


def train_fasttext_on_file(path, dim=300, epoch=5, minn=3, maxn=6, min_count=1, model_type='skipgram'):
    logging.info(f"Training FastText on {path} (dim={dim}, epoch={epoch})")
    # If the fasttext CLI binary is available, prefer running it so we can stream
    # its textual progress output to the console. Otherwise fall back to the
    # Python API (wrapped with a spinner).
    fasttext_cli = shutil.which('fasttext')
    if fasttext_cli:
        out_dir = tempfile.mkdtemp(prefix='ft_model_')
        out_prefix = os.path.join(out_dir, 'model')
        cmd = [
            fasttext_cli,
            model_type,
            '-input', path,
            '-output', out_prefix,
            '-dim', str(dim),
            '-epoch', str(epoch),
            '-minn', str(minn),
            '-maxn', str(maxn),
            '-minCount', str(min_count),
        ]
        logging.info(f"Running fastText CLI: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # Stream lines and show them so user sees progress info
        try:
            for raw in iter(proc.stdout.readline, b''):
                if not raw:
                    break
                line = raw.decode('utf-8', errors='ignore').rstrip()
                # Print progress lines without interfering with tqdm bars
                tqdm.write(line)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"fasttext CLI failed with return code {proc.returncode}")
            model_path = out_prefix + '.bin'
            if not os.path.exists(model_path):
                raise RuntimeError(f"Expected fasttext model at {model_path} not found")
            model = fasttext.load_model(model_path)
        finally:
            # cleanup temporary files (keep model files if caller wanted to save explicit paths)
            try:
                # remove temporary model files
                for fn in os.listdir(out_dir):
                    os.remove(os.path.join(out_dir, fn))
                os.rmdir(out_dir)
            except Exception:
                pass
        return model

    # Fallback: use Python API in background and show a spinner
    def _train():
        return fasttext.train_unsupervised(
            input=path,
            model=model_type,
            dim=dim,
            epoch=epoch,
            minn=minn,
            maxn=maxn,
            minCount=min_count,
            verbose=0
        )

    model = _run_with_spinner(_train, desc=f"Training FastText ({os.path.basename(path)})")
    return model


def build_embeddings_for_vocab(fasttext_model, vocab_list, clean_fn=None, dim=None):
    """Return numpy array of shape (len(vocab_list), dim)."""
    if dim is None:
        dim = fasttext_model.get_dimension()
    embs = np.zeros((len(vocab_list), dim), dtype=np.float32)
    for i, token in enumerate(tqdm(vocab_list, desc='Building embeddings')):
        t = token
        if clean_fn:
            t = clean_fn(token)
        # fasttext returns vector even for OOV (via n-grams)
        try:
            vec = fasttext_model.get_word_vector(t)
        except Exception:
            vec = np.zeros(dim, dtype=np.float32)
        embs[i] = vec
    return embs


def compute_cost_matrix(A, B, eps=1e-8, block_size=1024):
    """Compute cost matrix C = 1 - cosine(A, B) where A: (n, d), B: (m, d).
    Compute in row-blocks and show a tqdm progress bar to indicate progress.
    Returns float32 matrix shape (n, m).
    """
    n, _ = A.shape
    m, _ = B.shape
    # normalize
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)

    # allocate output
    C = np.empty((n, m), dtype=np.float32)

    # process in blocks of rows to provide progress updates
    with tqdm(total=n, desc='Computing cost matrix (rows)', unit='rows') as p:
        for start in range(0, n, block_size):
            end = min(n, start + block_size)
            # compute block dot
            block = 1.0 - np.dot(A_norm[start:end], B_norm.T)
            C[start:end] = block.astype(np.float32)
            p.update(end - start)

    return C


def run_ot(teacher_vocab_size, student_vocab_size, cost_matrix, ot_reg=0.1, ot_numitermax=1000):
    a = np.ones(teacher_vocab_size, dtype=np.float64) / float(teacher_vocab_size)
    b = np.ones(student_vocab_size, dtype=np.float64) / float(student_vocab_size)
    logging.info("Solving Sinkhorn OT (this may take some time)")
    # Run ot.sinkhorn in background and show spinner so user sees progress
    def _do_sinkhorn():
        return ot.sinkhorn(a, b, cost_matrix, reg=ot_reg, numItermax=ot_numitermax)

    P = _run_with_spinner(_do_sinkhorn, desc='Running Sinkhorn OT')
    return P


def main():
    parser = argparse.ArgumentParser(description="Create global alignment matrix via corpus-based FastText + OT")
    parser.add_argument('--teacher-model', default='bert-base-uncased', help='teacher tokenizer/model name or path')
    parser.add_argument('--teacher-tokenizer', default=None, help='(Optional) explicit tokenizer repo/path for teacher (use when teacher arg points to adapter)')
    parser.add_argument('--student-model', default='distilbert-base-uncased', help='student tokenizer/model name or path')
    parser.add_argument('--student-tokenizer', default=None, help='(Optional) explicit tokenizer repo/path for student')
    parser.add_argument('--corpus-path', default=None, help='Path to large corpus (plain text, one document per line). If omitted the script will download and extract enwiki dump.')
    parser.add_argument('--output-path', default='./global_alignment.npy', help='Output .npy file for global alignment matrix')
    parser.add_argument('--teacher-fasttext-path', default=None, help='Optional path to save teacher fasttext .bin')
    parser.add_argument('--student-fasttext-path', default=None, help='Optional path to save student fasttext .bin')
    parser.add_argument('--fasttext-dim', type=int, default=100)
    parser.add_argument('--fasttext-epoch', type=int, default=5)
    parser.add_argument('--fasttext-minn', type=int, default=3)
    parser.add_argument('--fasttext-maxn', type=int, default=6)
    parser.add_argument('--fasttext-min-count', type=int, default=1)
    parser.add_argument('--ot-reg', type=float, default=0.1)
    parser.add_argument('--ot-numitermax', type=int, default=1000)
    parser.add_argument('--max-lines', type=int, default=None, help='Max lines from corpus to tokenize (for quick testing)')
    parser.add_argument('--top-k', type=int, default=None, help='Limit vocab to top-k most frequent tokens (per tokenizer) to reduce cost matrix size')
    parser.add_argument('--force', action='store_true', help='Overwrite existing output file')
    parser.add_argument('--allow-large', action='store_true', help='Allow constructing large cost matrices when system memory appears sufficient (uses /proc/meminfo to check). Use with caution.')
    parser.add_argument('--cleanup-extracted', action='store_true', help='Remove the large extracted wiki text file after tokenization to save disk space')
    parser.add_argument('--tmp-dir', default=None, help='Directory to store temporary working files (defaults to the directory of --output-path)')
    args = parser.parse_args()

    if os.path.exists(args.output_path) and (not args.force):
        logging.error(f"Output path {args.output_path} already exists. Use --force to overwrite.")
        sys.exit(1)

    # If no corpus provided, download and extract enwiki dump
    temp_download_bz2 = None
    temp_extracted_txt = None
    if args.corpus_path is None:
        # Use persistent wiki dataset directory requested by user
        tmp_root = '/mnt/hungpv/projects/MoEmb/dataset/wiki'
        os.makedirs(tmp_root, exist_ok=True)
        dump_url = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2'
        temp_download_bz2 = os.path.join(tmp_root, 'enwiki-latest-pages-articles.xml.bz2')
        temp_extracted_txt = os.path.join(tmp_root, 'enwiki_extracted.txt')

        # If extracted file already exists and not forcing, reuse it
        if os.path.exists(temp_extracted_txt) and (not args.force):
            logging.info(f"Found existing extracted wiki at {temp_extracted_txt}, reusing (use --force to re-download)")
            args.corpus_path = temp_extracted_txt
        else:
            try:
                # download only if needed
                if (not os.path.exists(temp_download_bz2)) or args.force:
                    download_file(dump_url, temp_download_bz2)
            except Exception as e:
                logging.error(f"Failed to download enwiki dump: {e}")
                sys.exit(1)

            # Try extracting with gensim, otherwise try WikiExtractor if installed
            try:
                # Attempt to stream-tokenize directly to avoid creating a huge extracted file
                logging.info("Attempting streaming extract + tokenize to avoid large intermediate file")
                # load tokenizers early for streaming
                teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_tokenizer or args.teacher_model)
                student_tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer or args.student_model)
                # create tmp dir for tokenized outputs
                tmp_base = args.tmp_dir or tmp_root
                os.makedirs(tmp_base, exist_ok=True)
                tmp_dir_local = tempfile.mkdtemp(prefix='align_stream_', dir=tmp_base)
                teacher_tok_path = os.path.join(tmp_dir_local, 'teacher_tokenized.txt')
                student_tok_path = os.path.join(tmp_dir_local, 'student_tokenized.txt')
                teacher_counter, student_counter = stream_tokenize_wiki_bz2(temp_download_bz2, teacher_tokenizer, student_tokenizer, teacher_tok_path, student_tok_path, max_pages=args.max_lines)
                # set tokenized paths for downstream steps and avoid creating extracted text
                args.corpus_path = None
                # set the tokenized files used later
                stream_used = True
            except Exception as e:
                logging.exception("Streaming tokenization failed, falling back to full extraction: %s", e)
                # fallback to full extraction if streaming fails
                try:
                    extract_wiki_bz2_to_text(temp_download_bz2, temp_extracted_txt)
                except Exception as e2:
                    logging.exception("gensim-based wiki extraction failed with exception")
                    logging.info("Attempting to fallback to WikiExtractor.py (if available in PATH)")
                    we_cmd = shutil.which('WikiExtractor.py') or shutil.which('wikiextractor') or shutil.which('WikiExtractor')
                    if we_cmd:
                        try:
                            logging.info(f"Running WikiExtractor: {we_cmd}")
                            we_out_dir = os.path.join(tmp_root, 'we_out')
                            os.makedirs(we_out_dir, exist_ok=True)
                            subprocess.check_call([we_cmd, '-b', '100M', '-o', we_out_dir, temp_download_bz2])
                            with open(temp_extracted_txt, 'w', encoding='utf-8') as fout:
                                all_files = []
                                for root, _, files in os.walk(we_out_dir):
                                    for fn in files:
                                        all_files.append(os.path.join(root, fn))
                                for path in tqdm(all_files, desc='Concatenating WikiExtractor outputs'):
                                    with open(path, 'r', encoding='utf-8', errors='ignore') as inf:
                                        shutil.copyfileobj(inf, fout)
                        except Exception as e:
                            logging.error(f"WikiExtractor fallback failed: {e}")
                            logging.error("Install gensim or WikiExtractor and try again.")
                            sys.exit(1)
                    else:
                        logging.error("Neither gensim extraction nor WikiExtractor are available/succeeded to extract enwiki.")
                        logging.error("If you have gensim installed, check the exception above. Otherwise install WikiExtractor (https://github.com/attardi/wikiextractor) or provide --corpus-path to a plain text corpus.")
                        sys.exit(1)
                args.corpus_path = temp_extracted_txt

    logging.info("Loading tokenizers...")

    # Teacher tokenizer: try explicit override first, then teacher_model, then progressively shorter hyphen-truncated names
    teacher_tokenizer = None
    teacher_try_names = []
    if args.teacher_tokenizer:
        teacher_try_names.append(args.teacher_tokenizer)
    teacher_try_names.append(args.teacher_model)
    if '-' in args.teacher_model:
        parts = args.teacher_model.split('-')
        for i in range(len(parts) - 1, 0, -1):
            candidate = '-'.join(parts[:i])
            if candidate not in teacher_try_names:
                teacher_try_names.append(candidate)

    last_err = None
    for name in teacher_try_names:
        try:
            teacher_tokenizer = AutoTokenizer.from_pretrained(name)
            logging.info(f"Loaded teacher tokenizer from: {name}")
            break
        except Exception as e:
            last_err = e
            logging.warning(f"Failed to load teacher tokenizer '{name}': {e}")
            continue
    if teacher_tokenizer is None:
        raise RuntimeError("Failed to load teacher tokenizer. Provide a valid tokenizer repo/path via --teacher-tokenizer or a model name with tokenizer/config.json.")

    # Student tokenizer: try explicit override first, then student_model
    if args.student_tokenizer:
        try:
            student_tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer)
            logging.info(f"Loaded student tokenizer from: {args.student_tokenizer}")
        except Exception as e:
            logging.warning(f"Failed to load student tokenizer override '{args.student_tokenizer}': {e}. Falling back to --student-model")
            student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    else:
        student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)

    # Tokenize corpus separately
    # Use a temp directory on the same filesystem as the output (likely /mnt) to avoid filling root (/)
    default_tmp_base = os.path.dirname(os.path.abspath(args.output_path)) or '/mnt/hungpv/projects/MoEmb/dataset/tmp'
    tmp_base = args.tmp_dir or default_tmp_base
    os.makedirs(tmp_base, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix='align_', dir=tmp_base)
    teacher_tok_path = os.path.join(tmp_dir, 'teacher_tokenized.txt')
    student_tok_path = os.path.join(tmp_dir, 'student_tokenized.txt')

    logging.info("Tokenizing corpus with teacher tokenizer...")
    teacher_counter = tokenize_corpus_to_file(teacher_tokenizer, args.corpus_path, teacher_tok_path, max_lines=args.max_lines, desc='Teacher tokenization')
    logging.info("Tokenizing corpus with student tokenizer...")
    student_counter = tokenize_corpus_to_file(student_tokenizer, args.corpus_path, student_tok_path, max_lines=args.max_lines, desc='Student tokenization')

    # Decide vocab lists (optionally top-k by frequency)
    teacher_vocab = list(teacher_tokenizer.get_vocab().keys())
    student_vocab = list(student_tokenizer.get_vocab().keys())

    # Clean function to match token strings used in tokenized corpus
    def clean_token(t):
        return t.replace('Ġ', '').replace('▁', '').strip()

    if args.top_k is not None:
        logging.info(f"Selecting top-{args.top_k} tokens by corpus frequency for each tokenizer")
        # Map vocab tokens to cleaned form and pick top-k by frequency from counter
        # Build frequency dict restricted to vocab cleaned forms
        teacher_freq = {tok: teacher_counter.get(clean_token(tok), 0) for tok in teacher_vocab}
        student_freq = {tok: student_counter.get(clean_token(tok), 0) for tok in student_vocab}

        # sort and select
        teacher_sorted = sorted(teacher_freq.items(), key=lambda x: x[1], reverse=True)
        student_sorted = sorted(student_freq.items(), key=lambda x: x[1], reverse=True)

        teacher_selected = [t for t, _ in teacher_sorted[:args.top_k]]
        student_selected = [t for t, _ in student_sorted[:args.top_k]]

        teacher_vocab_sel = teacher_selected
        student_vocab_sel = student_selected
    else:
        teacher_vocab_sel = teacher_vocab
        student_vocab_sel = student_vocab

    logging.info(f"Teacher vocab size selected: {len(teacher_vocab_sel)}")
    logging.info(f"Student vocab size selected: {len(student_vocab_sel)}")

    # Safety check for cost matrix size. Instead of a fixed cutoff, estimate required bytes and
    # check available system memory (if readable from /proc/meminfo). The user can override
    # with --allow-large to attempt construction even when the static heuristic would fail.
    prod = int(len(teacher_vocab_sel)) * int(len(student_vocab_sel))
    required_bytes = prod * 4  # float32
    # Try to read available memory from /proc/meminfo (Linux). Fallback to None.
    mem_available_bytes = None
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    parts = line.split()
                    # value is in kB
                    mem_available_bytes = int(parts[1]) * 1024
                    break
    except Exception:
        mem_available_bytes = None

    # Conservative extra factor to account for other allocations during computation
    safety_factor = 1.6
    est_needed = int(required_bytes * safety_factor)

    # Do a more nuanced check using MemAvailable if possible. Fall back to requiring
    # --allow-large when meminfo is not readable and the matrix is very large.
    if prod > 0:
        if mem_available_bytes is not None:
            logging.info(f"Estimated cost matrix size: {required_bytes / (1024**3):.2f} GiB, with safety factor: {(est_needed) / (1024**3):.2f} GiB")
            logging.info(f"System reported MemAvailable: {mem_available_bytes / (1024**3):.2f} GiB")
            if mem_available_bytes < est_needed:
                if args.allow_large:
                    logging.warning("MemAvailable appears smaller than estimated required memory, but --allow-large was passed. Proceeding (may fail).")
                else:
                    logging.error("Not enough available memory to safely construct the full cost matrix. Use --top-k to reduce vocabulary sizes or rerun with --allow-large to force attempt.")
                    sys.exit(1)
        else:
            # Could not read meminfo; require explicit allow for very large matrices
            if prod > 5e7 and not args.allow_large:
                logging.error("Could not detect available system memory (unable to read /proc/meminfo). For very large matrices use --allow-large to bypass this check or use --top-k to reduce sizes.")
                sys.exit(1)
            if args.allow_large:
                logging.warning("/proc/meminfo not readable but --allow-large passed; proceeding to attempt building large cost matrix (may fail).")

    # Train FastText models on tokenized files
    teacher_ft = train_fasttext_on_file(teacher_tok_path, dim=args.fasttext_dim, epoch=args.fasttext_epoch, minn=args.fasttext_minn, maxn=args.fasttext_maxn, min_count=args.fasttext_min_count)
    student_ft = train_fasttext_on_file(student_tok_path, dim=args.fasttext_dim, epoch=args.fasttext_epoch, minn=args.fasttext_minn, maxn=args.fasttext_maxn, min_count=args.fasttext_min_count)

    if args.teacher_fasttext_path:
        teacher_ft.save_model(args.teacher_fasttext_path)
        logging.info(f"Saved teacher fasttext model to {args.teacher_fasttext_path}")
    if args.student_fasttext_path:
        student_ft.save_model(args.student_fasttext_path)
        logging.info(f"Saved student fasttext model to {args.student_fasttext_path}")

    # Build embeddings for selected vocab
    logging.info("Building embeddings for selected vocabularies...")
    teacher_embs = build_embeddings_for_vocab(teacher_ft, teacher_vocab_sel, clean_fn=clean_token, dim=args.fasttext_dim)
    student_embs = build_embeddings_for_vocab(student_ft, student_vocab_sel, clean_fn=clean_token, dim=args.fasttext_dim)

    logging.info(f"Teacher embeddings shape: {teacher_embs.shape}")
    logging.info(f"Student embeddings shape: {student_embs.shape}")

    # Compute cost matrix
    cost = compute_cost_matrix(teacher_embs, student_embs)
    logging.info(f"Cost matrix shape: {cost.shape}")

    # Run OT
    ot_matrix = run_ot(len(teacher_vocab_sel), len(student_vocab_sel), cost, ot_reg=args.ot_reg, ot_numitermax=args.ot_numitermax)

    # Save output (also save meta mapping lists)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    np.save(args.output_path, ot_matrix.astype(np.float32))
    meta_path = args.output_path + '.meta.npz'
    np.savez_compressed(meta_path, teacher_vocab=teacher_vocab_sel, student_vocab=student_vocab_sel)
    logging.info(f"Saved global alignment matrix to {args.output_path} and vocab metadata to {meta_path}")

    # Cleanup temporary files
    try:
        os.remove(teacher_tok_path)
        os.remove(student_tok_path)
        os.rmdir(tmp_dir)
    except Exception:
        pass

    logging.info('Done')


if __name__ == '__main__':
    main()
