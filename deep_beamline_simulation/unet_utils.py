# import statements
import torch
import numpy as np
from torchinfo import summary
from deep_beamline_simulation.u_net import Block, Encoder, Decoder, UNet


def shape_handling():
    # Check the shape of each block of the unet depending on the input
    block = Block(1, 64)
    # input shape
    x = torch.randn(1, 1, 128, 128)
    # output shape
    print("Block Shape:")
    print(block(x).shape)

    # check the shape of the encoder block depending on the inputs
    encoder = Encoder()
    # input image
    x = torch.randn(1, 3, 128, 128)
    encoder_features = encoder(x)
    print("Encoder Output Shape:")
    for feature in encoder_features:
        print(feature.shape)

    decoder = Decoder()
    x = torch.randn(1, 128, 28, 28)
    print("Decoder Output Shape:")
    print(decoder(x, ftrs[::-1][1:]).shape)


def model_summary():
    model = UNet()
    print("Model Structure")
    print(model)
    print("Model Summary")
    summary(model)


def single_input():
    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_func = torch.nn.MSELoss()
    # (number of lists, sub lists, num rows, num columns)
    inputs = torch.randn(1, 3, 20, 20)
    outputs = torch.randn(1, 1, 4, 4)
    test_in = torch.randn(2, 3, 20, 20)
    test_out = torch.randn(2, 1, 4, 4)
    # sanity check
    # print(model(inputs).shape)
    # print(outputs.shape)
    for e in range(0, 10):
        predictions = model(inputs)
        loss = loss_func(predictions, outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Test Loss: " + str(loss))
    test_predictions = None
    with torch.no_grad():
        model.eval()
        test_predictions = model(test_in)
        test_loss = loss_func(test_predictions, test_out)
        test_plot = test_predictions.data.numpy()[0]
    print("Test Loss: " + str(test_loss))
    print(test_predictions)


def unet_dataloader():
    random_dataset = []
    for i in range(0, 100):
        random_dataset.append(
            (
                torch.from_numpy(np.random.rand(3, 20, 20).astype("f")),
                (torch.from_numpy(np.random.rand(1, 20, 20).astype("f"))),
            )
        )
    random_dataset_loader = DataLoader(random_dataset, batch_size=10, shuffle=True)
    # Training
    model = UNet()
    print(model)
    summary(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_func = torch.nn.MSELoss()

    epochs = 5
    loss_list = []
    for e in range(epochs):
        training_loss = 0.0
        for img, label in random_dataset_loader:
            prediction = model(img)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.data.item()
            loss_list.append(training_loss)
        if e % 10 == 0:
            print("Epoch: {}, Training Loss: {:.2f}".format(e, training_loss))

    # plt.plot(loss_list)
    # plt.show()
