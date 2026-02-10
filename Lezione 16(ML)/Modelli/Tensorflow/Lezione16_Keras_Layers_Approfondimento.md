# Lezione 16 — Keras / TensorFlow: Lista “completa” dei layer + opzioni principali (GitHub)

Questa è la versione **da GitHub** della risposta che ti ho dato in chat (con dettagli e link concettuali).  
Se alcune parti si ripetono rispetto al file precedente, va bene: qui è “tutto in uno”.

---

## 1) Premessa: cosa significa “lista completa” in Keras

Keras (e `tf.keras`) contiene **molti** layer e alcuni cambiano/si aggiungono in base alla versione (Keras 3 vs tf.keras).  
Per questo ti do:

1. **Un catalogo completo per categorie** (quali layer esistono e quali opzioni principali hanno).
2. Uno **script** che genera automaticamente la lista completa con **tutte le firme** (tutti gli argomenti reali) per **la tua versione installata**.

Questo è il modo più “preciso” possibile, perché evita errori e differenze tra versioni.

---

## 2) Parametri “base” comuni a (quasi) tutti i layer

Questi parametri di solito sono presenti come argomenti espliciti o dentro `**kwargs`:

- `name`: nome del layer (utile per debugging e per cercarlo nel grafico del modello)
- `dtype`: tipo numerico (es. `float32`)
- `trainable`: se `False`, i pesi del layer non vengono aggiornati durante l’allenamento
- altri parametri di backend (dipendono da Keras / tf.keras)

**Cosa imparare bene qui:** `trainable=False` è importantissimo quando fai transfer learning, encoder congelati, ecc.

---

## 3) Core layers (fondamentali in MLP, AE, VAE, FFN dei Transformer)

### Layer inclusi
- `Input` / `keras.Input(...)` (crea un “tensore simbolico” di input)
- `Dense`
- `EinsumDense`
- `Activation`
- `Embedding`
- `Masking`
- `Lambda`
- `Identity`

### Opzioni principali per ciascuno (quelle che ti servono davvero)

**Input**
- `shape`: forma dell’input (senza batch)
- `batch_size`: (opzionale)
- `dtype`, `name`

**Dense**
- `units`: numero neuroni (dimensione output)
- `activation`: es. `'relu'`, `'softmax'`, `None`
- `use_bias`: `True/False`
- `kernel_initializer`, `bias_initializer`
- `kernel_regularizer`, `bias_regularizer`, `activity_regularizer`
- `kernel_constraint`, `bias_constraint`

**Embedding**
- `input_dim`: dimensione vocabolario
- `output_dim`: dimensione embedding
- `mask_zero`: se `True`, crea una maschera per padding (super utile per NLP)
- `embeddings_initializer`, `embeddings_regularizer`, `embeddings_constraint`

**Lambda**
- `function`: la funzione Python da applicare
- `output_shape`: se non inferibile
- `mask`: gestione maschera
- `arguments`: argomenti extra

---

## 4) Convolution layers (CNN — immagini e feature extractor per GAN)

### Layer inclusi
- `Conv1D`, `Conv2D`, `Conv3D`
- `SeparableConv1D`, `SeparableConv2D`
- `DepthwiseConv1D`, `DepthwiseConv2D`
- `Conv1DTranspose`, `Conv2DTranspose`, `Conv3DTranspose`

### Opzioni principali (quasi sempre presenti)
- `filters` (numero canali output) *(tranne alcuni depthwise)*
- `kernel_size` (dimensione filtro)
- `strides`
- `padding`: `'valid'` o `'same'`
- `dilation_rate`
- `activation`
- `use_bias`
- `data_format`: `'channels_last'` o `'channels_first'`
- initializer/regularizer/constraint come in Dense

---

## 5) Pooling layers (riduzione dimensionale / invariance)

### Layer inclusi
- `MaxPooling1D/2D/3D`
- `AveragePooling1D/2D/3D`
- `GlobalMaxPooling1D/2D/3D`
- `GlobalAveragePooling1D/2D/3D`
- `AdaptiveAveragePooling1D/2D/3D`
- `AdaptiveMaxPooling1D/2D/3D`

### Opzioni principali
- pooling classico: `pool_size`, `strides`, `padding`, `data_format`
- global pooling: spesso `data_format`, `keepdims`
- adaptive pooling: parametri “target size” (varia a seconda del layer)

---

## 6) Recurrent layers (sequenze: LSTM/GRU — utili anche se poi userai Transformer)

### Layer inclusi
- `SimpleRNN`, `SimpleRNNCell`
- `LSTM`, `LSTMCell`
- `GRU`, `GRUCell`
- `BaseRNN`, `StackedRNNCells`
- `TimeDistributed`, `Bidirectional`
- `ConvLSTM1D/2D/3D`

### Opzioni principali (quelle che ti devi ricordare)
- `units`
- `activation`, `recurrent_activation`
- `return_sequences`: `True` se vuoi output per ogni timestep
- `return_state`: `True` se vuoi stati finali
- `dropout`, `recurrent_dropout` (valori in `[0, 1]`)
- `stateful`: mantiene stato fra batch (usare con attenzione)
- `go_backwards`, `unroll`
- initializer/regularizer/constraint

---

## 7) Attention layers (il cuore dei Transformer)

### Layer inclusi
- `MultiHeadAttention`
- `GroupQueryAttention`
- `Attention`
- `AdditiveAttention`

### Opzioni principali (Transformer)
**MultiHeadAttention**
- `num_heads`
- `key_dim`
- `value_dim` (opzionale)
- `dropout`
- `attention_axes`
- `use_bias`
- `output_shape` (opzionale)
- initializer/regularizer/constraint

---

## 8) Normalization layers (stabilità e performance)

### Layer inclusi
- `BatchNormalization`
- `LayerNormalization`
- `UnitNormalization`
- `GroupNormalization`
- `RMSNormalization`

### Opzioni principali
- `axis`
- `epsilon`
- BatchNorm: `momentum`, `center`, `scale`
- altri: `center`, `scale` (quando presenti)

---

## 9) Regularization layers (overfitting e robustezza)

### Layer inclusi
- `Dropout`
- `SpatialDropout1D/2D/3D`
- `GaussianDropout`
- `AlphaDropout`
- `GaussianNoise`
- `ActivityRegularization`

### Opzioni principali
- Dropout: `rate`, `noise_shape`, `seed`
- GaussianNoise: `stddev`, `seed`
- ActivityRegularization: `l1`, `l2`

---

## 10) Reshaping layers (cambia forma dei tensori)

### Layer inclusi
- `Reshape`, `Flatten`
- `RepeatVector`, `Permute`
- `Cropping1D/2D/3D`
- `UpSampling1D/2D/3D`
- `ZeroPadding1D/2D/3D`

### Opzioni principali
- `target_shape` (Reshape)
- `data_format` dove rilevante
- upsampling: `size` (+ a volte `interpolation`)
- cropping/padding: valori specifici per dimensione

---

## 11) Merging layers (somma, concatenazione, ecc.)

### Layer inclusi
- `Concatenate`
- `Add`, `Subtract`, `Multiply`
- `Average`, `Maximum`, `Minimum`
- `Dot`

### Opzioni principali
- `axis` (Concatenate)
- `axes`/`normalize` (Dot)

---

## 12) Activation layers (funzioni di attivazione “a layer”)

### Layer inclusi
- `ReLU`
- `Softmax`
- `LeakyReLU`
- `PReLU`
- `ELU`

### Opzioni principali
- `ReLU(max_value, negative_slope, threshold)`
- `Softmax(axis=...)`
- `LeakyReLU(negative_slope=...)`
- `PReLU` (parametri/constraint su `alpha`)
- `ELU(alpha=...)`

---

## 13) Backend-specific layers (wrappers per altri backend)

- `TorchModuleWrapper`
- `TensorflowSavedModelLayer`
- `JaxLayer`
- `FlaxLayer`

Opzioni: dipendono dal wrapper, perché “incapsulano” moduli esterni.

---

## 14) Preprocessing layers (super importanti in progetti reali)

Questi layer servono per “portare dentro” nel modello preprocessing/augmentation.

### Testo
- `TextVectorization`  
  Opzioni principali: `standardize`, `split`, `ngrams`, `output_mode`, `output_sequence_length`, vocabolario, ecc.

### Numerico
- `Normalization`
- `Discretization`
- (altri in base a versione)

### Categoriale
- `CategoryEncoding`
- `Hashing`
- `HashedCrossing`
- `StringLookup`
- `IntegerLookup`

### Immagini — preprocessing
- `Resizing`, `Rescaling`, `CenterCrop`, `AutoContrast`, ecc.

### Immagini — augmentation
- `RandomFlip`, `RandomRotation`, `RandomZoom`, `RandAugment`, `MixUp`, `CutMix`, `AugMix`, ecc.

### Audio
- `STFTSpectrogram`, `MelSpectrogram` (dove presenti)

---

# 15) Metodo perfetto: genera “lista completa + TUTTE le opzioni” automaticamente

## Perché ti serve
Perché Keras cambia, ed è facile sbagliare una firma.

## Script: genera tutte le firme dei layer in un file Markdown

Salvalo come `genera_firme_layer.py` e lancialo con:
```bash
python genera_firme_layer.py
```

```python
import inspect
from pathlib import Path

def get_layers_module():
    # Prova prima Keras "standalone" (Keras 3)
    try:
        import keras
        from keras import layers
        return "keras", layers
    except Exception:
        # Fallback: TensorFlow Keras
        import tensorflow as tf
        return "tf.keras", tf.keras.layers

def is_layer_class(obj, base_layer):
    return isinstance(obj, type) and issubclass(obj, base_layer)

def main(out_path="KERAS_LAYERS_SIGNATURES.md"):
    prefix, layers_mod = get_layers_module()
    base = layers_mod.Layer

    items = []
    for name, obj in vars(layers_mod).items():
        if is_layer_class(obj, base):
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "(signature non disponibile)"
            items.append((name, sig))

    items.sort(key=lambda x: x[0].lower())

    lines = [f"# Firme costruttori layer ({prefix})", ""]
    for name, sig in items:
        lines.append(f"## {name}")
        if prefix == "tf.keras":
            lines.append(f"`tf.keras.layers.{name}{sig}`")
        else:
            lines.append(f"`keras.layers.{name}{sig}`")
        lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Creato: {out_path} (layer trovati: {len(items)})")

if __name__ == "__main__":
    main()
```

### Cosa ottieni
- Un file `KERAS_LAYERS_SIGNATURES.md` con la **firma completa** di ogni layer
- Quindi: per ogni layer vedi **tutti gli argomenti** reali disponibili nella tua installazione

---

## 16) Come usare questa cosa per diventare autonomo su VAE/GAN/Transformer

- **Autoencoder/VAE**: impari bene `Dense`, `Conv2D`, `Conv2DTranspose`, `Sampling (Lambda)`, loss multiple.
- **GAN**: impari `Conv2D`, `BatchNorm`, `LeakyReLU`, e training loop custom.
- **Transformer**: impari `Embedding`, `MultiHeadAttention`, `LayerNormalization`, residual e maschere.

Questo catalogo + lo script ti danno **la visione completa**: categorie + layer + firme esatte.

