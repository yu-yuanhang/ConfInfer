/*
 * Copyright (c) 2016-2017, Linaro Limited
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef TA_CONFINFER_H
#define TA_CONFINFER_H

#include <confinfer_protocol.h>

/*
 * This UUID is generated with uuidgen
 * the ITU-T UUID generator at http://www.itu.int/ITU-T/asn1/uuid.html
 */
#define TA_CONFINFER_UUID \
	{ 0x7dd54ee6, 0x2f13, 0x4f1d, \
		{ 0xa3, 0xe6, 0x0f, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f} }

/* The function IDs implemented in this TA */
#define TA_CONFINFER_CMD_REGISTER_MODEL          CONFINFER_CMD_REGISTER_MODEL
#define TA_CONFINFER_CMD_LOAD_PARAMS             CONFINFER_CMD_LOAD_PARAMS
#define TA_CONFINFER_CMD_REGISTER_PARTITION      CONFINFER_CMD_REGISTER_PARTITION
#define TA_CONFINFER_CMD_EXEC_PARTITION          CONFINFER_CMD_EXEC_PARTITION
#define TA_CONFINFER_CMD_UNLOAD_MODEL            CONFINFER_CMD_UNLOAD_MODEL

/* Legacy debug / smoke commands */
#define TA_CONFINFER_CMD_INC_VALUE               CONFINFER_CMD_DEBUG_INC_VALUE
#define TA_CONFINFER_CMD_DEC_VALUE               CONFINFER_CMD_DEBUG_DEC_VALUE

#endif /* TA_CONFINFER_H */
