from torch.nn import CrossEntropyLoss
from torch.nn.modules import loss
from utils.TripletLoss import TripletLoss

class Loss(loss._Loss):
    def __init__(self, triplet_margin=1.2, l_triplet=1, l_xentropy=2, **kwargs):
        super(Loss, self).__init__()
        self.triplet = 0
        self.cross_entropy = 0
        self.total = 0
        self.cross_entropy_loss = CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=triplet_margin)

        self.total_loss = lambda triplet, xentropy: (l_triplet * triplet) + (l_xentropy * xentropy)

    def forward(self, outputs, labels):

        triplet = [self.triplet_loss(output, labels) for output in outputs[1:4]]
        self.triplet = sum(triplet) / len(triplet)

        cross_entropy = [self.cross_entropy_loss(output, labels) for output in outputs[4:]]
        self.cross_entropy = sum(cross_entropy) / len(cross_entropy)

        loss_sum = self.total_loss(self.triplet, self.cross_entropy)
        self.total = loss_sum

        return loss_sum
