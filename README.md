Optimal Music Recognition (OMR) is a specialized domain within computer vision focused on the interpretation
and conversion of sheet music images into usable musical notation. Current OMR methodologies primarily leverage Convolutional Recurrent Neural Networks (CRNNs)
or Convolutional Neural Networks (CNNs) combined with
transformer encoder-decoder architectures to generate sequences of musical symbols from input images.
However, with the advent of the Vision Transformer
(ViT), as introduced in ”An Image is Worth 16x16 Words”
, recent research indicates that minimizing the implicit bias
in deep learning models can enhance interpretative accuracy. Therefore, we explore a purely transformer-based approach to OMR, employing a pretrained ViT alongside a
transformer decoder to generate the desired musical symbol sequences.
Additionally, we incorporated an explicitly defined semantic musical vocabulary tailored for the transformer
encoder-decoder model. Despite encountering technical
challenges that prevented the complete training of our
model, we are confident that this approach represents the
future of efficient and accurate musical symbol recognition
from sheet music images.
