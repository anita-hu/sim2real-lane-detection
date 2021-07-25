"""credit: https://github.com/HsinYingLee/DRIT/blob/master/src/model.py
"""

from networks import drit_networks as networks
from collections import namedtuple
import torch
import torch.nn as nn

EncResult = namedtuple("EncResult", (
    "z_content_a", "z_content_b", "z_attr_a", "z_attr_b", "mu_a", "logvar_a", "mu_b", "logvar_b"
), defaults=(None, None, None, None))
ForwardResult = namedtuple("ForwardResult", (
    "fake_A_encoded",
    "fake_B_encoded",
    "fake_AA_encoded",
    "fake_BB_encoded",
    "fake_A_random",
    "fake_B_random",
    "fake_A_recon",
    "fake_B_recon",
    "z_random",
    "enc_result",
    "fake_A_random2",
    "fake_B_random2",
    "z_random2"
), defaults=(None, None, None))

ImageDisplay = namedtuple("ImageDisplay", (
    "real_A_encoded",
    "fake_B_encoded",
    "fake_B_random",
    "fake_AA_encoded",
    "fake_A_recon",
    "real_B_encoded",
    "fake_A_encoded",
    "fake_A_random",
    "fake_BB_encoded",
    "fake_B_recon"))


class DRIT_trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(DRIT, self).__init__()

        # parameters
        lr = hyperparameters['lr']
        lr_dcontent = lr / 2.5
        self.nz = 8
        self.concat = hyperparameters["concat"]
        self.no_ms = hyperparameters["no_ms"]

        # discriminators
        if hyperparameters["dis_scale"] > 1:
            self.disA = networks.MultiScaleDis(
                hyperparameters["input_dim_a"], hyperparameters["dis_scale"], norm=hyperparameters["dis_norm"], sn=hyperparameters["dis_spectral_norm"])
            self.disB = networks.MultiScaleDis(
                hyperparameters["input_dim_b"], hyperparameters["dis_scale"], norm=hyperparameters["dis_norm"], sn=hyperparameters["dis_spectral_norm"])
            self.disA2 = networks.MultiScaleDis(
                hyperparameters["input_dim_a"], hyperparameters["dis_scale"], norm=hyperparameters["dis_norm"], sn=hyperparameters["dis_spectral_norm"])
            self.disB2 = networks.MultiScaleDis(
                hyperparameters["input_dim_b"], hyperparameters["dis_scale"], norm=hyperparameters["dis_norm"], sn=hyperparameters["dis_spectral_norm"])
        else:
            self.disA = networks.Dis(
                hyperparameters["input_dim_a"], norm=hyperparameters["dis_norm"], sn=hyperparameters["dis_spectral_norm"])
            self.disB = networks.Dis(
                hyperparameters["input_dim_b"], norm=hyperparameters["dis_norm"], sn=hyperparameters["dis_spectral_norm"])
            self.disA2 = networks.Dis(
                hyperparameters["input_dim_a"], norm=hyperparameters["dis_norm"], sn=hyperparameters["dis_spectral_norm"])
            self.disB2 = networks.Dis(
                hyperparameters["input_dim_b"], norm=hyperparameters["dis_norm"], sn=hyperparameters["dis_spectral_norm"])
        self.disContent = networks.Dis_content()

        # encoders
        self.enc_c = networks.E_content(hyperparameters["input_dim_a"], hyperparameters["input_dim_b"])
        if self.concat:
            self.enc_a = networks.E_attr_concat(hyperparameters["input_dim_a"], hyperparameters["input_dim_b"], self.nz,
                                                norm_layer=None, nl_layer=networks.get_non_linearity(layer_type='lrelu'))
        else:
            self.enc_a = networks.E_attr(
                hyperparameters["input_dim_a"], hyperparameters["input_dim_b"], self.nz)

        # generator
        if self.concat:
            self.gen = networks.G_concat(
                hyperparameters["input_dim_a"], hyperparameters["input_dim_b"], nz=self.nz)
        else:
            self.gen = networks.G(
                hyperparameters["input_dim_a"], hyperparameters["input_dim_b"], nz=self.nz)

        # optimizers
        self.disA_opt = torch.optim.Adam(
            self.disA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disB_opt = torch.optim.Adam(
            self.disB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disA2_opt = torch.optim.Adam(
            self.disA2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disB2_opt = torch.optim.Adam(
            self.disB2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disContent_opt = torch.optim.Adam(self.disContent.parameters(
        ), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_c_opt = torch.optim.Adam(
            self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_a_opt = torch.optim.Adam(
            self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.gen_opt = torch.optim.Adam(
            self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

        # Setup the loss function for training
        self.criterionL1 = torch.nn.L1Loss()

    def initialize(self):
        self.disA.apply(networks.gaussian_weights_init)
        self.disB.apply(networks.gaussian_weights_init)
        self.disA2.apply(networks.gaussian_weights_init)
        self.disB2.apply(networks.gaussian_weights_init)
        self.disContent.apply(networks.gaussian_weights_init)
        self.gen.apply(networks.gaussian_weights_init)
        self.enc_c.apply(networks.gaussian_weights_init)
        self.enc_a.apply(networks.gaussian_weights_init)

    def set_scheduler(self, hyperparameters, last_ep=0):
        self.disA_sch = networks.get_scheduler(self.disA_opt, hyperparameters, last_ep)
        self.disB_sch = networks.get_scheduler(self.disB_opt, hyperparameters, last_ep)
        self.disA2_sch = networks.get_scheduler(self.disA2_opt, hyperparameters, last_ep)
        self.disB2_sch = networks.get_scheduler(self.disB2_opt, hyperparameters, last_ep)
        self.disContent_sch = networks.get_scheduler(
            self.disContent_opt, hyperparameters, last_ep)
        self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, hyperparameters, last_ep)
        self.enc_a_sch = networks.get_scheduler(self.enc_a_opt, hyperparameters, last_ep)
        self.gen_sch = networks.get_scheduler(self.gen_opt, hyperparameters, last_ep)

    def setgpu(self, gpu):
        self.gpu = gpu
        self.disA.cuda(self.gpu)
        self.disB.cuda(self.gpu)
        self.disA2.cuda(self.gpu)
        self.disB2.cuda(self.gpu)
        self.disContent.cuda(self.gpu)
        self.enc_c.cuda(self.gpu)
        self.enc_a.cuda(self.gpu)
        self.gen.cuda(self.gpu)

    def get_z_random(self, batchSize, nz, random_type='gauss'):
        z = torch.randn(batchSize, nz).cuda(self.gpu)
        return z

    def test_forward(self, image, a2b=True):
        z_random = self.get_z_random(image.size(0), self.nz, 'gauss')
        if a2b:
            z_content = self.enc_c.forward_a(image)
            output = self.gen.forward_b(z_content, z_random)
        else:
            z_content = self.enc_c.forward_b(image)
            output = self.gen.forward_a(z_content, z_random)
        return output

    def test_forward_transfer(self, image_a, image_b, a2b=True):
        z_content_a, z_content_b = self.enc_c.forward(image_a, image_b)
        if self.concat:
            mu_a, logvar_a, mu_b, logvar_b = self.enc_a.forward(
                image_a, image_b)
            std_a = logvar_a.mul(0.5).exp()
            eps = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
            z_attr_a = eps.mul(std_a).add(mu_a)
            std_b = logvar_b.mul(0.5).exp()
            eps = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
            z_attr_b = eps.mul(std_b).add(mu_b)
        else:
            z_attr_a, z_attr_b = self.enc_a.forward(image_a, image_b)
        if a2b:
            output = self.gen.forward_b(z_content_a, z_attr_b)
        else:
            output = self.gen.forward_a(z_content_b, z_attr_a)
        return output

    def encode(self, real_A_encoded, real_B_encoded) -> EncResult:
        # get encoded z_c
        z_content_a, z_content_b = self.enc_c.forward(
            real_A_encoded, real_B_encoded)

        # get encoded z_a
        if self.concat:
            mu_a, logvar_a, mu_b, logvar_b = self.enc_a.forward(
                real_A_encoded, real_B_encoded)
            std_a = logvar_b.mul(0.5).exp()
            eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
            z_attr_a = eps_a.mul(std_a).add(mu_a)
            std_b = logvar_b.mul(0.5).exp()
            eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
            z_attr_b = eps_b.mul(std_b).add(mu_b)
            return EncResult(z_content_a, z_content_b, z_attr_a, z_attr_b, mu_a, logvar_a, mu_b, logvar_b)
        else:
            z_attr_a, z_attr_b = self.enc_a.forward(
                real_A_encoded, real_B_encoded)
            return EncResult(z_content_a, z_content_b, z_attr_a, z_attr_b)

    def forward(self, real_A_encoded, real_B_encoded) -> ForwardResult:
        # get encoded z_c, z_a
        enc_result = self.encode(real_A_encoded, real_B_encoded)

        # get random z_a
        z_random = self.get_z_random(real_A_encoded.size(0), self.nz, 'gauss')
        if not self.no_ms:
            z_random2 = self.get_z_random(
                real_A_encoded.size(0), self.nz, 'gauss')

        # first cross translation
        if not self.no_ms:
            input_content_forA = torch.cat(
                (enc_result.z_content_b, enc_result.z_content_a, enc_result.z_content_b, enc_result.z_content_b), 0)
            input_content_forB = torch.cat(
                (enc_result.z_content_a, enc_result.z_content_b, enc_result.z_content_a, enc_result.z_content_a), 0)
            input_attr_forA = torch.cat(
                (enc_result.z_attr_a, enc_result.z_attr_a, z_random, z_random2), 0)
            input_attr_forB = torch.cat(
                (enc_result.z_attr_b, enc_result.z_attr_b, z_random, z_random2), 0)
        else:
            input_content_forA = torch.cat(
                (enc_result.z_content_b, enc_result.z_content_a, enc_result.z_content_b), 0)
            input_content_forB = torch.cat(
                (enc_result.z_content_a, enc_result.z_content_b, enc_result.z_content_a), 0)
            input_attr_forA = torch.cat(
                (enc_result.z_attr_a, enc_result.z_attr_a, z_random), 0)
            input_attr_forB = torch.cat(
                (enc_result.z_attr_b, enc_result.z_attr_b, z_random), 0)

        output_fakeA = self.gen.forward_a(input_content_forA, input_attr_forA)
        output_fakeB = self.gen.forward_b(input_content_forB, input_attr_forB)

        if not self.no_ms:
            fake_A_encoded, fake_AA_encoded, fake_A_random, fake_A_random2 = torch.split(
                output_fakeA, enc_result.z_content_a.size(0), dim=0)
            fake_B_encoded, fake_BB_encoded, fake_B_random, fake_B_random2 = torch.split(
                output_fakeB, enc_result.z_content_a.size(0), dim=0)
        else:
            fake_A_encoded, fake_AA_encoded, fake_A_random = torch.split(
                output_fakeA, enc_result.z_content_a.size(0), dim=0)
            fake_B_encoded, fake_BB_encoded, fake_B_random = torch.split(
                output_fakeB, enc_result.z_content_a.size(0), dim=0)

        # get reconstructed encoded z_c, z_a
        enc_result_recon = self.encode(fake_A_encoded, fake_B_encoded)

        # second cross translation
        fake_A_recon = self.gen.forward_a(
            enc_result_recon.z_content_a, enc_result_recon.z_attr_a)
        fake_B_recon = self.gen.forward_b(
            enc_result_recon.z_content_b, enc_result_recon.z_attr_b)

        if not self.no_ms:
            return ForwardResult(
                fake_A_encoded,
                fake_B_encoded,
                fake_AA_encoded,
                fake_BB_encoded,
                fake_A_random,
                fake_B_random,
                fake_A_recon,
                fake_B_recon,
                z_random,
                enc_result,
                fake_A_random2,
                fake_B_random2,
                z_random2
            )
        else:
            return ForwardResult(
                fake_A_encoded,
                fake_B_encoded,
                fake_AA_encoded,
                fake_BB_encoded,
                fake_A_random,
                fake_B_random,
                fake_A_recon,
                fake_B_recon,
                z_random,
                enc_result,
            )

    def forward_content(self, image_a, image_b):
        half_size = 1
        self.real_A_encoded = image_a[0:half_size]
        self.real_B_encoded = image_b[0:half_size]
        # get encoded z_c
        self.z_content_a, self.z_content_b = self.enc_c.forward(
            self.real_A_encoded, self.real_B_encoded)

    def update_D_content(self, image_a, image_b):
        self.forward_content(image_a, image_b)
        self.disContent_opt.zero_grad()
        loss_D_Content = self.backward_contentD(
            self.z_content_a, self.z_content_b)
        self.disContent_loss = loss_D_Content.item()
        nn.utils.clip_grad_norm(self.disContent.parameters(), 5)
        self.disContent_opt.step()

    def dis_update(self, image_a, image_b):
        # input images
        half_size = 1
        real_A_encoded = image_a[0:half_size]
        real_A_random = image_a[half_size:]
        real_B_encoded = image_b[0:half_size]
        real_B_random = image_b[half_size:]
        forward_results = self.forward(real_A_encoded, real_B_encoded)

        # for display
        self.image_display = ImageDisplay(
            real_A_encoded,
            forward_results.fake_B_encoded,
            forward_results.fake_B_random,
            forward_results.fake_AA_encoded,
            forward_results.fake_A_recon,
            real_B_encoded,
            forward_results.fake_A_encoded,
            forward_results.fake_A_random,
            forward_results.fake_BB_encoded,
            forward_results.fake_B_recon
        )

        # update disA
        self.disA_opt.zero_grad()
        loss_D1_A = self.backward_D(
            self.disA, real_A_encoded, forward_results.fake_A_encoded)
        self.disA_loss = loss_D1_A.item()
        self.disA_opt.step()

        # update disA2
        self.disA2_opt.zero_grad()
        loss_D2_A = self.backward_D(
            self.disA2, real_A_random, forward_results.fake_A_random)
        self.disA2_loss = loss_D2_A.item()
        if not self.no_ms:
            loss_D2_A2 = self.backward_D(
                self.disA2, real_A_random, forward_results.fake_A_random2)
            self.disA2_loss += loss_D2_A2.item()
        self.disA2_opt.step()

        # update disB
        self.disB_opt.zero_grad()
        loss_D1_B = self.backward_D(
            self.disB, real_B_encoded, forward_results.fake_B_encoded)
        self.disB_loss = loss_D1_B.item()
        self.disB_opt.step()

        # update disB2
        self.disB2_opt.zero_grad()
        loss_D2_B = self.backward_D(
            self.disB2, real_B_random, forward_results.fake_B_random)
        self.disB2_loss = loss_D2_B.item()
        if not self.no_ms:
            loss_D2_B2 = self.backward_D(
                self.disB2, real_B_random, forward_results.fake_B_random2)
            self.disB2_loss += loss_D2_B2.item()
        self.disB2_opt.step()

        # update disContent
        self.disContent_opt.zero_grad()
        loss_D_Content = self.backward_contentD(
            forward_results.enc_result.z_content_a, forward_results.enc_result.z_content_b)
        self.disContent_loss = loss_D_Content.item()
        nn.utils.clip_grad_norm(self.disContent.parameters(), 5)
        self.disContent_opt.step()

    def backward_D(self, netD, real, fake):
        pred_fake = netD.forward(fake.detach())
        pred_real = netD.forward(real)
        loss_D = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all0 = torch.zeros_like(out_fake).cuda(self.gpu)
            all1 = torch.ones_like(out_real).cuda(self.gpu)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            loss_D += ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D

    def backward_contentD(self, imageA, imageB):
        pred_fake = self.disContent.forward(imageA.detach())
        pred_real = self.disContent.forward(imageB.detach())
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all1 = torch.ones((out_real.size(0))).cuda(self.gpu)
            all0 = torch.zeros((out_fake.size(0))).cuda(self.gpu)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
        loss_D = ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D

    def gen_update(self, image_a, image_b):
        # input images
        half_size = 1
        real_A_encoded = image_a[0:half_size]
        real_B_encoded = image_b[0:half_size]
        forward_results = self.forward(real_A_encoded, real_B_encoded)

        # update G, Ec, Ea
        self.enc_c_opt.zero_grad()
        self.enc_a_opt.zero_grad()
        self.gen_opt.zero_grad()
        self.backward_EG(forward_results, real_A_encoded, real_B_encoded)
        self.enc_c_opt.step()
        self.enc_a_opt.step()
        self.gen_opt.step()

        forward_results = self.forward(real_A_encoded, real_B_encoded)

        # update G, Ec
        self.enc_c_opt.zero_grad()
        self.gen_opt.zero_grad()
        self.backward_G_alone(forward_results)
        self.enc_c_opt.step()
        self.gen_opt.step()

    def backward_EG(self, forward_results, real_A_encoded, real_B_encoded):
        # content Ladv for generator
        loss_G_GAN_Acontent = self.backward_G_GAN_content(
            forward_results.enc_result.z_content_a)
        loss_G_GAN_Bcontent = self.backward_G_GAN_content(
            forward_results.enc_result.z_content_b)

        # Ladv for generator
        loss_G_GAN_A = self.backward_G_GAN(
            forward_results.fake_A_encoded, self.disA)
        loss_G_GAN_B = self.backward_G_GAN(
            forward_results.fake_B_encoded, self.disB)

        # KL loss - z_a
        if self.concat:
            kl_element_a = forward_results.enc_result.mu_a.pow(2).add(
                forward_results.enc_result.logvar_a.exp()).mul(-1).add(1).add(forward_results.enc_result.logvar_a)
            loss_kl_za_a = torch.sum(kl_element_a).mul(-0.5) * 0.01
            kl_element_b = forward_results.enc_result.mu_b.pow(2).add(
                forward_results.enc_result.logvar_b.exp()).mul(-1).add(1).add(forward_results.enc_result.logvar_b)
            loss_kl_za_b = torch.sum(kl_element_b).mul(-0.5) * 0.01
        else:
            loss_kl_za_a = self._l2_regularize(
                forward_results.enc_result.z_attr_a) * 0.01
            loss_kl_za_b = self._l2_regularize(
                forward_results.enc_result.z_attr_b) * 0.01

        # KL loss - z_c
        loss_kl_zc_a = self._l2_regularize(
            forward_results.enc_result.z_content_a) * 0.01
        loss_kl_zc_b = self._l2_regularize(
            forward_results.enc_result.z_content_b) * 0.01

        # cross cycle consistency loss
        loss_G_L1_A = self.criterionL1(
            forward_results.fake_A_recon, real_A_encoded) * 10
        loss_G_L1_B = self.criterionL1(
            forward_results.fake_B_recon, real_B_encoded) * 10
        loss_G_L1_AA = self.criterionL1(
            forward_results.fake_AA_encoded, real_A_encoded) * 10
        loss_G_L1_BB = self.criterionL1(
            forward_results.fake_BB_encoded, real_B_encoded) * 10

        loss_G = loss_G_GAN_A + loss_G_GAN_B + \
            loss_G_GAN_Acontent + loss_G_GAN_Bcontent + \
            loss_G_L1_AA + loss_G_L1_BB + \
            loss_G_L1_A + loss_G_L1_B + \
            loss_kl_zc_a + loss_kl_zc_b + \
            loss_kl_za_a + loss_kl_za_b

        loss_G.backward()

        self.gan_loss_a = loss_G_GAN_A.item()
        self.gan_loss_b = loss_G_GAN_B.item()
        self.gan_loss_acontent = loss_G_GAN_Acontent.item()
        self.gan_loss_bcontent = loss_G_GAN_Bcontent.item()
        self.kl_loss_za_a = loss_kl_za_a.item()
        self.kl_loss_za_b = loss_kl_za_b.item()
        self.kl_loss_zc_a = loss_kl_zc_a.item()
        self.kl_loss_zc_b = loss_kl_zc_b.item()
        self.l1_recon_A_loss = loss_G_L1_A.item()
        self.l1_recon_B_loss = loss_G_L1_B.item()
        self.l1_recon_AA_loss = loss_G_L1_AA.item()
        self.l1_recon_BB_loss = loss_G_L1_BB.item()
        self.G_loss = loss_G.item()

    def backward_G_GAN_content(self, data):
        outs = self.disContent.forward(data)
        for out in outs:
            outputs_fake = torch.sigmoid(out)
            all_half = 0.5*torch.ones((outputs_fake.size(0))).cuda(self.gpu)
            ad_loss = nn.functional.binary_cross_entropy(
                outputs_fake, all_half)
        return ad_loss

    def backward_G_GAN(self, fake, netD=None):
        outs_fake = netD.forward(fake)
        loss_G = 0
        for out_a in outs_fake:
            outputs_fake = torch.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
            loss_G += nn.functional.binary_cross_entropy(
                outputs_fake, all_ones)
        return loss_G

    def backward_G_alone(self, forward_results):
        # Ladv for generator
        loss_G_GAN2_A = self.backward_G_GAN(
            forward_results.fake_A_random, self.disA2)
        loss_G_GAN2_B = self.backward_G_GAN(
            forward_results.fake_B_random, self.disB2)
        if not self.no_ms:
            loss_G_GAN2_A2 = self.backward_G_GAN(
                forward_results.fake_A_random2, self.disA2)
            loss_G_GAN2_B2 = self.backward_G_GAN(
                forward_results.fake_B_random2, self.disB2)

        # mode seeking loss for A-->B and B-->A
        if not self.no_ms:
            lz_AB = torch.mean(torch.abs(forward_results.fake_B_random2 - forward_results.fake_B_random)) / \
                torch.mean(torch.abs(forward_results.z_random2 -
                           forward_results.z_random))
            lz_BA = torch.mean(torch.abs(forward_results.fake_A_random2 - forward_results.fake_A_random)) / \
                torch.mean(torch.abs(forward_results.z_random2 -
                           forward_results.z_random))
            eps = 1 * 1e-5
            loss_lz_AB = 1 / (lz_AB + eps)
            loss_lz_BA = 1 / (lz_BA + eps)
        # latent regression loss
        # run self.forward before this
        if self.concat:
            mu2_a, _, mu2_b, _ = self.enc_a.forward(
                forward_results.fake_A_random, forward_results.fake_B_random)
            loss_z_L1_a = torch.mean(
                torch.abs(mu2_a - forward_results.z_random)) * 10
            loss_z_L1_b = torch.mean(
                torch.abs(mu2_b - forward_results.z_random)) * 10
        else:
            z_attr_random_a, z_attr_random_b = self.enc_a.forward(
                forward_results.fake_A_random, forward_results.fake_B_random)
            loss_z_L1_a = torch.mean(
                torch.abs(z_attr_random_a - forward_results.z_random)) * 10
            loss_z_L1_b = torch.mean(
                torch.abs(z_attr_random_b - forward_results.z_random)) * 10

        loss_z_L1 = loss_z_L1_a + loss_z_L1_b + loss_G_GAN2_A + loss_G_GAN2_B
        if not self.no_ms:
            loss_z_L1 += (loss_G_GAN2_A2 + loss_G_GAN2_B2)
            loss_z_L1 += (loss_lz_AB + loss_lz_BA)
        loss_z_L1.backward()
        self.l1_recon_z_loss_a = loss_z_L1_a.item()
        self.l1_recon_z_loss_b = loss_z_L1_b.item()
        if not self.no_ms:
            self.gan2_loss_a = loss_G_GAN2_A.item() + loss_G_GAN2_A2.item()
            self.gan2_loss_b = loss_G_GAN2_B.item() + loss_G_GAN2_B2.item()
            self.lz_AB = loss_lz_AB.item()
            self.lz_BA = loss_lz_BA.item()
        else:
            self.gan2_loss_a = loss_G_GAN2_A.item()
            self.gan2_loss_b = loss_G_GAN2_B.item()

    def update_learning_rate(self):
        self.disA_sch.step()
        self.disB_sch.step()
        self.disA2_sch.step()
        self.disB2_sch.step()
        self.disContent_sch.step()
        self.enc_c_sch.step()
        self.enc_a_sch.step()
        self.gen_sch.step()

    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir)
        # weight
        if train:
            self.disA.load_state_dict(checkpoint['disA'])
            self.disA2.load_state_dict(checkpoint['disA2'])
            self.disB.load_state_dict(checkpoint['disB'])
            self.disB2.load_state_dict(checkpoint['disB2'])
            self.disContent.load_state_dict(checkpoint['disContent'])
        self.enc_c.load_state_dict(checkpoint['enc_c'])
        self.enc_a.load_state_dict(checkpoint['enc_a'])
        self.gen.load_state_dict(checkpoint['gen'])
        # optimizer
        if train:
            self.disA_opt.load_state_dict(checkpoint['disA_opt'])
            self.disA2_opt.load_state_dict(checkpoint['disA2_opt'])
            self.disB_opt.load_state_dict(checkpoint['disB_opt'])
            self.disB2_opt.load_state_dict(checkpoint['disB2_opt'])
            self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
            self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
            self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        return checkpoint['ep'], checkpoint['total_it']

    def save(self, filename, ep, total_it):
        state = {
            'disA': self.disA.state_dict(),
            'disA2': self.disA2.state_dict(),
            'disB': self.disB.state_dict(),
            'disB2': self.disB2.state_dict(),
            'disContent': self.disContent.state_dict(),
            'enc_c': self.enc_c.state_dict(),
            'enc_a': self.enc_a.state_dict(),
            'gen': self.gen.state_dict(),
            'disA_opt': self.disA_opt.state_dict(),
            'disA2_opt': self.disA2_opt.state_dict(),
            'disB_opt': self.disB_opt.state_dict(),
            'disB2_opt': self.disB2_opt.state_dict(),
            'disContent_opt': self.disContent_opt.state_dict(),
            'enc_c_opt': self.enc_c_opt.state_dict(),
            'enc_a_opt': self.enc_a_opt.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, filename)
        return

    def assemble_outputs(self):
        images_a = self.normalize_image(
            self.image_display.real_A_encoded).detach()
        images_b = self.normalize_image(
            self.image_display.real_B_encoded).detach()
        images_a1 = self.normalize_image(
            self.image_display.fake_A_encoded).detach()
        images_a2 = self.normalize_image(
            self.image_display.fake_A_random).detach()
        images_a3 = self.normalize_image(
            self.image_display.fake_A_recon).detach()
        images_a4 = self.normalize_image(
            self.image_display.fake_AA_encoded).detach()
        images_b1 = self.normalize_image(
            self.image_display.fake_B_encoded).detach()
        images_b2 = self.normalize_image(
            self.image_display.fake_B_random).detach()
        images_b3 = self.normalize_image(
            self.image_display.fake_B_recon).detach()
        images_b4 = self.normalize_image(
            self.image_display.fake_BB_encoded).detach()
        row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::],
                         images_b2[0:1, ::], images_a4[0:1, ::], images_a3[0:1, ::]), 3)
        row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::],
                         images_a2[0:1, ::], images_b4[0:1, ::], images_b3[0:1, ::]), 3)
        return torch.cat((row1, row2), 2)

    def get_image_display(self):
        return torch.cat((self.image_display.real_A_encoded[0:1].detach().cpu(), self.image_display.fake_B_encoded[0:1].detach().cpu(),
                          self.image_display.fake_B_random[0:1].detach().cpu(), self.image_display.fake_AA_encoded[0:1].detach(
        ).cpu(), self.image_display.fake_A_recon[0:1].detach().cpu(),
            self.image_display.real_B_encoded[0:1].detach().cpu(
        ), self.image_display.fake_A_encoded[0:1].detach().cpu(),
            self.image_display.fake_A_random[0:1].detach().cpu(), self.image_display.fake_BB_encoded[0:1].detach().cpu(), self.image_display.fake_B_recon[0:1].detach().cpu()), dim=0)

    def normalize_image(self, x):
        return x[:, 0:3, :, :]

    def sample(self, x_a, x_b):
        """
        x_a and x_b contains multiple images
        """
        self.eval()
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            forward_result = self.forward(x_a[i], x_b[i])
            x_a_recon.append(forward_result.fake_A_recon)
            x_b_recon.append(forward_result.fake_B_recon)
            x_ba1.append(forward_result.fake_A_random)
            x_ba2.append(forward_result.fake_A_random2)
            x_ab1.append(forward_result.fake_B_random)
            x_ab2.append(forward_result.fake_B_random2)
            
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2