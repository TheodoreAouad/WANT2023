from .bimonn import BiMoNN


class BimonnIdentity(BiMoNN):

    def forward(self, x):
        return x
