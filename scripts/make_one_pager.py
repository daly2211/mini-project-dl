"""Build a simple one page PDF summary of the mini-project."""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate

OUT = Path(__file__).resolve().parents[1] / "report" / "one_pager.pdf"

LINK = "#1A4FD6"

TITLE = ParagraphStyle(
    "title",
    fontName="Helvetica-Bold",
    fontSize=16,
    leading=19.5,
    spaceAfter=2.5 * mm,
)
SUB = ParagraphStyle(
    "sub",
    fontName="Helvetica",
    fontSize=10,
    leading=13.5,
    spaceAfter=6.0 * mm,
)
BODY = ParagraphStyle(
    "body",
    fontName="Helvetica",
    fontSize=9.4,
    leading=13.0,
    spaceAfter=3.3 * mm,
)


def link(url, text):
    return f'<link href="{url}" color="{LINK}">{text}</link>'


P = []

P.append(Paragraph(
    "When a CNN learns the background instead of the object",
    TITLE,
))
P.append(Paragraph(
    "Mini-project, RNA and Deep Learning, Master 2 SPI. Theme: Projet 4, Explainable AI.",
    SUB,
))

P.append(Paragraph(
    "Some image classifiers learn to cheat. In the dataset I use, almost every picture of a "
    "waterbird also contains water, and almost every picture of a landbird also contains land. "
    "A model can pick up that pattern and answer from the background alone, without really "
    "looking at the object. This works well until the test data breaks the pattern, for example "
    "a waterbird standing on land, and then the model fails. Researchers call this a shortcut, "
    "and it is one reason why a high average score can hide a serious weakness.",
    BODY,
))

P.append(Paragraph(
    "My question is whether the explanation of a prediction can warn us about this. I use "
    "Grad-CAM, a method that produces a heat map showing which regions of the image pushed the "
    "prediction. If the heat map sits mostly on the background and not on the object, then the "
    "model probably used the background. The question for the whole project is therefore "
    "simple: does a CNN make more mistakes when its Grad-CAM explanation focuses on the "
    "background?",
    BODY,
))

P.append(Paragraph(
    "Three papers frame the work. " + link("https://arxiv.org/abs/1911.08731", "Sagawa et al. (2020)")
    + " built the dataset I use and the four groups that come with it. They show that a normal "
    "model can reach a high average score and still do very badly on the rare groups, and their "
    "fix is stronger regularisation with early stopping, which is a change in training rather "
    "than in the data. " + link("https://arxiv.org/abs/1610.02391", "Selvaraju et al. (2017)")
    + " built Grad-CAM, the method I use. It is cheap and needs no retraining, but the maps are "
    "small and therefore approximate, and the authors also claim that Grad-CAM helps to reveal "
    "bias in a dataset. My last experiment tests that claim directly, and for me it did not "
    "work. " + link("https://arxiv.org/abs/2006.09994", "Xiao et al. (2021)")
    + " study the same background problem on ImageNet. They show that a model can classify an "
    "image from the background alone and that changing the background causes up to 87.5 percent "
    "of the errors, which is the closest result to my question, although they modify the image "
    "while I only read the explanation.",
    BODY,
))

P.append(Paragraph(
    "For the experiment I take the official Waterbirds split, with 4,795 images for training, "
    "1,199 for validation and 5,794 for testing. Each image has a bird label, landbird or "
    "waterbird, and a background label, land or water, and the group is a combination of the "
    "two. I train a ResNet18 pretrained on ImageNet with a new last layer of two classes, using "
    "AdamW with a learning rate of 1e-4, a batch size of 32, at most 8 epochs, early stopping "
    "on the validation score and a fixed seed of 42. Images are resized to 224 pixels and "
    "normalised with the ImageNet values. I also load the bird boxes, because they let me "
    "measure where the model looks. The measure is the share of the Grad-CAM map that falls "
    "outside the bird box, so a high share means the model looked mostly outside the bird.",
    BODY,
))

P.append(Paragraph(
    "The overall test accuracy is 87.76 percent, but the groups are very different. Landbird on "
    "land reaches 99.4 percent, landbird on water 87.3 percent, waterbird on land only 47.7 "
    "percent, and waterbird on water 88.8 percent, so the average hides a large failure on one "
    "group, exactly as Sagawa et al. describe. For my question the numbers are clear: on a "
    "correct prediction the share outside the box is 40.00 percent, and on a wrong prediction "
    "it is 69.07 percent, a gap of 29 points, with a correlation of minus 0.363 between that "
    "share and a correct answer. When the explanation looks mostly at the background, the answer "
    "is much more often wrong. I stay careful about the wording, because this is a link and not "
    "a proof of cause.",
    BODY,
))

P.append(Paragraph(
    "My own idea was to use the explanation to act on the training data. I ran the baseline "
    "Grad-CAM on the training set and selected the 25 percent of images where the model looks "
    "most outside the box, then trained the same network again with stronger augmentation on "
    "those images only, using bigger crops, colour jitter and random erasing. Every other "
    "setting stayed the same and the test set was never used to choose the images. The result is "
    "negative: the overall accuracy stayed at 87.76 percent, the worst group fell from 47.66 to "
    "40.03 percent and the mean share outside the box rose from 43.55 to 45.48 percent. Only "
    "landbird on water improved, by 2.8 points. My reading is that a big crop sometimes removes "
    "the bird, so the model learns to use the context even more, which is also why Sagawa et al. "
    "insist that the real fix belongs to the training objective.",
    BODY,
))

P.append(Paragraph(
    "In short, Grad-CAM gives a real and measurable warning: looking at the background goes "
    "together with making a mistake. It is useful for a user who wants to know when to trust a "
    "prediction, but it does not prove the cause, and in my hands it was not enough to improve "
    "the model. The main limits are that the boxes only approximate the object, that I used one "
    "seed and one model, and that I never tested whether the explanation itself is faithful. The "
    "next steps would be to change the training objective instead of the data, to use real "
    "background masks, and to repeat the runs with more seeds.",
    BODY,
))

doc = SimpleDocTemplate(
    str(OUT),
    pagesize=A4,
    leftMargin=19 * mm,
    rightMargin=19 * mm,
    topMargin=16 * mm,
    bottomMargin=14 * mm,
    title="When a CNN learns the background instead of the object",
    author="RNA et Deep Learning, Master 2 SPI",
)
doc.build(P)
print("wrote", OUT)
